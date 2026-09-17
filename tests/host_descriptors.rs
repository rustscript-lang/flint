use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use flint_ai::{
    FROZEN_RUSTSCRIPT_REV, flint_host_catalog, flint_host_modules, install_flint_host_modules,
};
use vm::{
    CompileSourceFileOptions, HostAdapterDescriptor, HostApiCatalog, HostBindingKind,
    HostFunctionRegistry, HostFunctionSchema, HostTypeSchema, SourceFlavor, Vm,
    compile_source_with_flavor_and_options,
};

fn compose_catalog() -> HostApiCatalog {
    flint_host_catalog().as_ref().clone()
}

fn schema_is_dynamic(schema: &HostTypeSchema) -> bool {
    match schema {
        HostTypeSchema::Unknown | HostTypeSchema::Map(_) => true,
        HostTypeSchema::Array(inner) | HostTypeSchema::Optional(inner) => schema_is_dynamic(inner),
        HostTypeSchema::Callable { params, result } => {
            params.iter().any(schema_is_dynamic) || schema_is_dynamic(result)
        }
        HostTypeSchema::Named { fields, .. } => {
            fields.iter().any(|field| schema_is_dynamic(&field.ty))
        }
        _ => false,
    }
}

fn dynamic_slots(catalog: &HostApiCatalog) -> Vec<String> {
    let mut dynamic = Vec::new();
    for function in catalog.functions() {
        if schema_is_dynamic(&function.return_type) {
            dynamic.push(format!("{} return", function.name));
        }
        for param in &function.params {
            if schema_is_dynamic(&param.ty) {
                dynamic.push(format!("{} param {}", function.name, param.name));
            }
        }
    }
    dynamic.sort();
    dynamic
}

fn rss_paths(dir: &str) -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dir);
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("rss"))
        .collect();
    paths.sort();
    paths
}

fn compile_rss(path: &Path) {
    flint_ai::compile_script_file(path)
        .unwrap_or_else(|err| panic!("{} failed to compile: {err}", path.display()));
}

fn is_residual_name(name: &str) -> bool {
    name.starts_with("flint::tokenizer::")
        || name.starts_with("flint::weights::")
        || name.starts_with("flint::tensor::")
        || name.starts_with("flint::nn::")
        || name.starts_with("flint::image::")
        || name.starts_with("flint::vl::")
}

fn compile_against(catalog: Arc<HostApiCatalog>, source: &str) -> vm::CompiledProgram {
    compile_source_with_flavor_and_options(
        source,
        SourceFlavor::RustScript,
        CompileSourceFileOptions::default().with_host_api_catalog(catalog),
    )
    .expect("script compiles")
}

#[test]
fn composition_catalog_and_adapter_names_are_the_same_surface() {
    let modules = flint_host_modules();
    let composition_names: Vec<String> = modules
        .iter()
        .flat_map(|module| module.descriptors())
        .map(|descriptor| descriptor.schema.name.clone())
        .collect();
    let adapter_names: Vec<String> = modules
        .iter()
        .flat_map(|module| module.descriptors())
        .map(|descriptor| match descriptor.adapter {
            HostAdapterDescriptor::StaticArgs(_)
            | HostAdapterDescriptor::StaticNonYieldingArgs(_)
            | HostAdapterDescriptor::Static(_)
            | HostAdapterDescriptor::StaticStack(_)
            | HostAdapterDescriptor::StaticStackRuntimeOwned(_)
            | HostAdapterDescriptor::Owned(_) => descriptor.schema.name.clone(),
        })
        .collect();
    let catalog_names: Vec<String> = flint_host_catalog()
        .functions()
        .iter()
        .map(|function| function.name.clone())
        .collect();

    let composition_count = composition_names.len();
    let adapter_count = adapter_names.len();
    let catalog_count = catalog_names.len();
    assert_eq!(composition_count, adapter_count);
    assert_eq!(adapter_count, catalog_count);
    assert_eq!(
        composition_names, catalog_names,
        "composition order must match catalog order"
    );

    let composition_set: BTreeSet<_> = composition_names.iter().cloned().collect();
    let adapter_set: BTreeSet<_> = adapter_names.iter().cloned().collect();
    let catalog_set: BTreeSet<_> = catalog_names.iter().cloned().collect();
    assert_eq!(composition_set.len(), composition_count);
    assert_eq!(composition_set, adapter_set);
    assert_eq!(adapter_set, catalog_set);

    let residual_count = composition_names
        .iter()
        .filter(|name| is_residual_name(name))
        .count();
    let macro_count = composition_count - residual_count;
    assert_eq!(macro_count, 105);
    assert_eq!(residual_count, 102);
    assert_eq!(composition_count, 207);

    for descriptor in modules.iter().flat_map(|module| module.descriptors()) {
        if !is_residual_name(&descriptor.schema.name) {
            continue;
        }
        assert_eq!(descriptor.binding.kind, HostBindingKind::StaticArgs);
        assert!(
            matches!(descriptor.adapter, HostAdapterDescriptor::StaticArgs(_)),
            "{} must bind a static args-slice adapter",
            descriptor.schema.name
        );
        assert!(descriptor.effects.is_empty());
        assert!(descriptor.resource_types.is_empty());
    }
}

#[test]
fn catalog_fingerprint_is_deterministic() {
    let first = compose_catalog();
    let second = compose_catalog();
    assert_eq!(first.fingerprint(), second.fingerprint());
    assert_eq!(first.fingerprint(), flint_host_catalog().fingerprint());
    let rendered = format!("{}", first.fingerprint());
    assert_eq!(rendered.len(), 16);
    let golden = include_str!("golden/flint_host_catalog_fingerprint.txt").trim();
    assert_eq!(rendered, golden);
}

#[test]
fn dynamic_map_any_unknown_slots_match_allowlist() {
    let catalog = flint_host_catalog();
    assert_eq!(
        dynamic_slots(catalog.as_ref()),
        ["flint::cli::get return", "flint::cli::refer param initial",]
    );
}

#[test]
fn deny_before_install_then_allow_after() {
    let catalog = flint_host_catalog();
    let compiled = compile_against(
        catalog.clone(),
        "use flint;\nlet _parser = flint::cli::argument_parser();\n",
    );

    let mut denied = Vm::new(compiled.program.clone());
    let empty = HostFunctionRegistry::restricted();
    let error = empty
        .bind_vm_cached(&mut denied)
        .expect_err("restricted bind must deny before install");
    let message = error.to_string();
    assert!(
        message.contains("flint::cli::argument_parser") || message.contains("not authorized"),
        "unexpected deny-before error: {message}"
    );

    let mut allowed = Vm::new(compiled.program);
    let mut registry = HostFunctionRegistry::restricted();
    install_flint_host_modules(&mut registry, catalog.as_ref()).expect("install composition");
    registry
        .bind_vm_cached(&mut allowed)
        .expect("allow after exact install");
}

#[test]
fn public_install_binds_the_complete_catalog_surface() {
    let catalog = flint_host_catalog();
    let mut registry = HostFunctionRegistry::restricted();
    install_flint_host_modules(&mut registry, catalog.as_ref()).expect("install composition");
    for function in catalog.functions() {
        assert!(
            registry.contains_name(&function.name),
            "install must register {}",
            function.name
        );
    }

    let compiled = compile_against(
        catalog.clone(),
        "use flint;\nlet _cleared = flint::tokenizer::clear_generated_tokens();\n",
    );
    let mut vm = Vm::new(compiled.program);
    registry
        .bind_vm_cached(&mut vm)
        .expect("public install helper must bind residual hosts");
}

#[test]
fn unlisted_host_import_is_denied_at_bind() {
    let production = flint_host_catalog();
    let mut builder = HostApiCatalog::builder();
    for function in production.functions() {
        builder.function(function.clone());
    }
    builder.function(HostFunctionSchema::with_return(
        "flint::not_exported",
        vec![],
        HostTypeSchema::Bool,
    ));
    let expanded = Arc::new(builder.build().expect("expanded catalog builds"));
    let compiled = compile_against(expanded, "use flint;\nlet _ok = flint::not_exported();\n");

    let mut registry = HostFunctionRegistry::restricted();
    install_flint_host_modules(&mut registry, production.as_ref()).expect("install composition");
    let mut vm = Vm::new(compiled.program);
    let error = registry
        .bind_vm_cached(&mut vm)
        .expect_err("unlisted import must be denied at bind");
    let message = error.to_string();
    assert!(
        message.contains("flint::not_exported"),
        "bind denial must name the unlisted import: {message}"
    );
}

#[test]
fn residual_schema_mismatch_rolls_back_after_earlier_modules() {
    let production = flint_host_catalog();
    let mut builder = HostApiCatalog::builder();
    for function in production.functions() {
        if function.name == "flint::vl::scatter_image_embeddings" {
            let mut mismatched = function.clone();
            mismatched.return_type = HostTypeSchema::String;
            builder.function(mismatched);
        } else {
            builder.function(function.clone());
        }
    }
    let mismatched = builder.build().expect("mismatched catalog builds");

    let mut registry = HostFunctionRegistry::restricted();
    let named_before = registry.named_struct_schemas().clone();
    install_flint_host_modules(&mut registry, &mismatched)
        .expect_err("residual adapter/schema mismatch must fail");
    assert_eq!(registry.named_struct_schemas(), &named_before);
    assert!(!registry.contains_name("flint::cli::argument_parser"));
    assert!(!registry.contains_name("flint::tokenizer::load"));
    assert!(!registry.contains_name("flint::vl::scatter_image_embeddings"));

    let compiled = compile_against(
        production.clone(),
        "use flint;\nlet _parser = flint::cli::argument_parser();\n",
    );
    let mut vm = Vm::new(compiled.program);
    registry
        .bind_vm_cached(&mut vm)
        .expect_err("failed residual install must leave capabilities unchanged");
}

#[test]
fn production_bind_path_has_no_parallel_registry_writer() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let host = fs::read_to_string(manifest.join("src/host.rs")).expect("host.rs");
    assert!(
        !host.contains("CONTEXT_HOST_OPS"),
        "TorchHostRuntime must not keep a parallel CONTEXT_HOST_OPS table"
    );
    assert!(
        !host.contains("register_catalog_args"),
        "production bind must not write the registry besides install_flint_host_modules"
    );
    assert!(
        !host.contains("bind_args_function"),
        "production bind must not post-bind args-slice adapters"
    );
    assert!(
        !manifest.join("src/host/context_schemas.rs").exists(),
        "parallel context_schemas.rs table must be deleted"
    );
    assert!(
        !host.contains("mod context_schemas"),
        "host.rs must not keep the parallel schema module"
    );
    assert!(
        host.contains("install_flint_host_modules"),
        "TorchHostRuntime must install through the public helper"
    );
}

#[test]
fn rustscript_crates_are_pinned_to_the_frozen_full_sha() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("Cargo.toml");
    let cargo_lock = fs::read_to_string(manifest_dir.join("Cargo.lock")).expect("Cargo.lock");
    assert!(
        cargo_toml.contains(FROZEN_RUSTSCRIPT_REV),
        "Cargo.toml must pin the frozen full SHA"
    );
    assert!(
        !cargo_toml.contains("path = \"../rustscript"),
        "production RustScript crates must not use sibling path deps"
    );
    let expected_source = format!(
        "git+https://github.com/rustscript-lang/rustscript?rev={FROZEN_RUSTSCRIPT_REV}#{FROZEN_RUSTSCRIPT_REV}"
    );
    for crate_name in ["pd-vm", "pd-host-function", "pd-host-schema"] {
        assert!(
            cargo_lock.contains(&format!("name = \"{crate_name}\"")),
            "{crate_name} must appear in Cargo.lock"
        );
        assert!(
            cargo_lock.contains(&expected_source),
            "{crate_name} must resolve to the frozen git SHA"
        );
    }
}

#[test]
fn bundled_rss_scripts_compile_through_the_production_catalog() {
    let scripts = rss_paths("scripts");
    let fixtures = rss_paths("tests/fixtures");
    assert!(
        !scripts.is_empty(),
        "expected bundled scripts under scripts/"
    );
    for path in scripts.into_iter().chain(fixtures) {
        compile_rss(&path);
    }
}

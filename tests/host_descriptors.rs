use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use flint_ai::{
    FROZEN_RUSTSCRIPT_REV, flint_host_catalog, flint_host_modules, install_flint_host_modules,
};
use vm::{
    CompileSourceFileOptions, HostApiCatalog, HostFunctionRegistry, HostTypeSchema, SourceFlavor,
    Vm, compile_source_with_flavor_and_options,
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

#[test]
fn composition_owns_every_migrated_host_in_declaration_order() {
    let modules = flint_host_modules();
    assert_eq!(
        modules.iter().map(|module| module.name).collect::<Vec<_>>(),
        [
            "flint.cli",
            "flint.runtime",
            "flint.cache",
            "flint.pair",
            "flint.ggml",
            "flint.llama",
            "flint.diffusion",
        ]
    );

    let names: Vec<String> = modules
        .iter()
        .flat_map(|module| module.descriptors())
        .map(|descriptor| descriptor.schema.name.clone())
        .collect();
    assert_eq!(names.len(), 105);

    let unique: BTreeSet<_> = names.iter().cloned().collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "host function names must be unique"
    );

    let catalog_names: Vec<String> = flint_host_catalog()
        .functions()
        .iter()
        .map(|function| function.name.clone())
        .collect();
    assert_eq!(&catalog_names[..names.len()], names.as_slice());
    assert!(
        catalog_names[names.len()..]
            .iter()
            .all(|name| name.starts_with("flint::tokenizer::")
                || name.starts_with("flint::weights::")
                || name.starts_with("flint::tensor::")
                || name.starts_with("flint::nn::")
                || name.starts_with("flint::image::")
                || name.starts_with("flint::vl::")),
        "residual catalog names must be the Torch CONTEXT_HOST_OPS families"
    );
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
        [
            "flint::cli::add_option param names",
            "flint::cli::get return",
            "flint::cli::refer param initial",
            "flint::runtime::args return",
        ]
    );
}

#[test]
fn deny_before_install_then_allow_after() {
    let catalog = flint_host_catalog();
    let compiled = compile_source_with_flavor_and_options(
        "use flint;\nlet _parser = flint::cli::argument_parser();\n",
        SourceFlavor::RustScript,
        CompileSourceFileOptions::default().with_host_api_catalog(catalog.clone()),
    )
    .expect("script compiles against production catalog");

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
    assert!(registry.contains_name("flint::cli::argument_parser"));
}

#[test]
fn unlisted_host_is_denied() {
    let catalog = flint_host_catalog();
    let mut registry = HostFunctionRegistry::restricted();
    install_flint_host_modules(&mut registry, catalog.as_ref()).expect("install composition");
    assert!(!registry.contains_name("flint::not_exported"));
}

#[test]
fn schema_mismatch_and_empty_catalog_do_not_partially_mutate() {
    let mut registry = HostFunctionRegistry::restricted();
    let empty = HostApiCatalog::builder()
        .build()
        .expect("empty catalog builds");
    install_flint_host_modules(&mut registry, &empty)
        .expect_err("install against an empty catalog must fail");
    assert!(
        !registry.contains_name("flint::cli::argument_parser"),
        "failed composition must not leave the first module installed"
    );

    let production = flint_host_catalog();
    let mut builder = HostApiCatalog::builder();
    for function in production.functions() {
        if function.name == "flint::cli::argument_parser" {
            let mut mismatched = function.clone();
            mismatched.return_type = HostTypeSchema::String;
            builder.function(mismatched);
        } else {
            builder.function(function.clone());
        }
    }
    let mismatched = builder.build().expect("mismatched catalog builds");
    install_flint_host_modules(&mut registry, &mismatched)
        .expect_err("schema/adapter mismatch must fail");
    assert!(
        !registry.contains_name("flint::cli::argument_parser"),
        "schema mismatch must not partially mutate the registry"
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

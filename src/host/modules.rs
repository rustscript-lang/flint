use std::sync::{Arc, OnceLock};

use vm::{
    HostApiCatalog, HostFunctionDescriptor, HostFunctionRegistry, HostModuleDescriptor, VmError,
    VmResult,
};

use super::{cache, cli, context, diffusion, ggml, llama, pair, runtime};

pub const FROZEN_RUSTSCRIPT_REV: &str = "b1d6cffede77f49410bf63525f30b9a46b02dc01";

/// Single ordered composition for every production Flint host.
///
/// Macro-generated descriptors and residual args-slice Torch adapters share this
/// list as the only compile-catalog and restricted-runtime authority.
///
/// Slot allowlist for remaining Any guest values:
/// - `flint::cli::get` / `flint::cli::refer` `Value` slots: argparse values are
///   string|int|float|bool chosen at refer time.
/// - llama/diffusion/cli/cache/pair/runtime `i64` handles: opaque tokens into
///   host-owned tables with explicit free; VM `HostResource` auto-close would
///   change llama.cpp/sd.cpp lifetime.
/// - ggml/llama device listings: formatted text reports consumed as `string`.
pub fn flint_host_modules() -> [HostModuleDescriptor; 13] {
    [
        cli_host_module(),
        runtime_host_module(),
        cache_host_module(),
        pair_host_module(),
        ggml_host_module(),
        llama_host_module(),
        diffusion_host_module(),
        context::tokenizer_host_module(),
        context::weights_host_module(),
        context::tensor_host_module(),
        context::nn_host_module(),
        context::image_host_module(),
        context::vl_host_module(),
    ]
}

fn cli_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.cli",
        functions: &[
            cli::cli_argument_parser_descriptor,
            cli::cli_set_description_descriptor,
            cli::cli_refer_descriptor,
            cli::cli_add_option_descriptor,
            cli::cli_add_argument_descriptor,
            cli::cli_required_descriptor,
            cli::cli_metavar_descriptor,
            cli::cli_parse_args_descriptor,
            cli::cli_get_descriptor,
        ],
        resources: &[],
    }
}

fn runtime_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.runtime",
        functions: &[
            runtime::runtime_args_descriptor,
            runtime::runtime_input_descriptor,
            runtime::runtime_arg_descriptor,
            runtime::runtime_arg_int_descriptor,
            runtime::runtime_arg_int_or_descriptor,
            runtime::runtime_arg_float_or_descriptor,
            runtime::runtime_arg_or_descriptor,
            runtime::runtime_set_output_descriptor,
            runtime::runtime_set_text_output_descriptor,
            runtime::runtime_start_timer_descriptor,
            runtime::runtime_start_decode_timer_descriptor,
            runtime::runtime_set_token_count_descriptor,
            runtime::runtime_set_decode_token_count_descriptor,
            runtime::runtime_compact2_descriptor,
        ],
        resources: &[],
    }
}

fn cache_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.cache",
        functions: &[
            cache::cache_clear_descriptor,
            cache::cache_has_descriptor,
            cache::cache_get_descriptor,
            cache::cache_set_descriptor,
        ],
        resources: &[],
    }
}

fn pair_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.pair",
        functions: &[
            pair::pair_new_descriptor,
            pair::pair_local_descriptor,
            pair::pair_global_descriptor,
        ],
        resources: &[],
    }
}

fn ggml_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.ggml",
        functions: &[
            ggml::ggml_load_backends_descriptor,
            ggml::ggml_list_devices_descriptor,
            ggml::ggml_stable_diffusion_package_dir_descriptor,
            ggml::ggml_load_stable_diffusion_backends_descriptor,
            ggml::ggml_list_stable_diffusion_devices_descriptor,
        ],
        resources: &[],
    }
}

fn llama_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.llama",
        functions: &[
            llama::llama_backend_init_descriptor,
            llama::llama_backend_supports_gpu_offload_descriptor,
            llama::llama_backend_list_devices_descriptor,
            llama::llama_backend_free_descriptor,
            llama::llama_model_params_init_descriptor,
            llama::llama_model_params_set_gpu_layers_descriptor,
            llama::llama_model_params_set_main_gpu_descriptor,
            llama::llama_model_params_set_memory_descriptor,
            llama::llama_model_load_descriptor,
            llama::llama_model_free_descriptor,
            llama::llama_model_n_ctx_train_descriptor,
            llama::llama_model_n_vocab_descriptor,
            llama::llama_model_tokenize_descriptor,
            llama::llama_model_is_eog_descriptor,
            llama::llama_chat_template_descriptor,
            llama::llama_chat_messages_init_descriptor,
            llama::llama_chat_messages_add_descriptor,
            llama::llama_apply_chat_template_descriptor,
            llama::llama_chat_free_descriptor,
            llama::llama_tokens_len_descriptor,
            llama::llama_tokens_get_descriptor,
            llama::llama_tokens_free_descriptor,
            llama::llama_context_params_init_descriptor,
            llama::llama_context_params_set_sizes_descriptor,
            llama::llama_context_params_set_threads_descriptor,
            llama::llama_context_new_descriptor,
            llama::llama_context_n_ctx_descriptor,
            llama::llama_context_decode_descriptor,
            llama::llama_context_free_descriptor,
            llama::llama_batch_init_descriptor,
            llama::llama_batch_add_descriptor,
            llama::llama_batch_add_sequence_descriptor,
            llama::llama_batch_clear_descriptor,
            llama::llama_batch_free_descriptor,
            llama::llama_sampler_chain_init_descriptor,
            llama::llama_sampler_add_top_k_descriptor,
            llama::llama_sampler_add_top_p_descriptor,
            llama::llama_sampler_add_min_p_descriptor,
            llama::llama_sampler_add_temp_descriptor,
            llama::llama_sampler_add_dist_descriptor,
            llama::llama_sampler_add_greedy_descriptor,
            llama::llama_sampler_chain_build_descriptor,
            llama::llama_sampler_sample_descriptor,
            llama::llama_sampler_accept_descriptor,
            llama::llama_sampler_free_descriptor,
            llama::llama_decoder_init_descriptor,
            llama::llama_decoder_push_descriptor,
            llama::llama_decoder_free_descriptor,
        ],
        resources: &[],
    }
}

fn diffusion_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.diffusion",
        functions: &[
            diffusion::sd_ctx_params_init_descriptor,
            diffusion::sd_ctx_params_set_paths_descriptor,
            diffusion::sd_ctx_params_set_backend_descriptor,
            diffusion::sd_ctx_params_set_wtype_descriptor,
            diffusion::sd_ctx_params_set_vae_format_descriptor,
            diffusion::sd_ctx_params_set_flags_descriptor,
            diffusion::sd_new_sd_ctx_descriptor,
            diffusion::sd_free_sd_ctx_descriptor,
            diffusion::sd_img_gen_params_init_descriptor,
            diffusion::sd_img_gen_params_set_prompt_descriptor,
            diffusion::sd_img_gen_params_set_size_descriptor,
            diffusion::sd_img_gen_params_set_sample_descriptor,
            diffusion::sd_img_gen_params_set_sampler_descriptor,
            diffusion::sd_str_to_sample_method_descriptor,
            diffusion::sd_str_to_scheduler_descriptor,
            diffusion::sd_sample_method_name_descriptor,
            diffusion::sd_scheduler_name_descriptor,
            diffusion::sd_get_default_sample_method_descriptor,
            diffusion::sd_get_default_scheduler_descriptor,
            diffusion::sd_generate_image_descriptor,
            diffusion::sd_images_save_descriptor,
            diffusion::sd_free_sd_images_descriptor,
        ],
        resources: &[],
    }
}

fn compose_flint_host_catalog() -> Result<HostApiCatalog, vm::HostApiCatalogError> {
    let descriptors: Vec<_> = flint_host_modules()
        .iter()
        .flat_map(|module| module.descriptors())
        .collect();
    HostFunctionDescriptor::collect_catalog(&descriptors)
}

/// Guest catalog derived from [`flint_host_modules`] in declaration order.
pub fn flint_host_catalog() -> Arc<HostApiCatalog> {
    static CATALOG: OnceLock<Arc<HostApiCatalog>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            Arc::new(
                compose_flint_host_catalog().expect("flint host composition must build a catalog"),
            )
        })
        .clone()
}

/// Installs every composed Flint module into `registry` against `catalog`.
///
/// The whole composition is transactional: a later module failure leaves
/// `registry` unchanged.
pub fn install_flint_host_modules(
    registry: &mut HostFunctionRegistry,
    catalog: &HostApiCatalog,
) -> VmResult<HostApiCatalog> {
    registry.transactionally(|registry| {
        let mut installed = None;
        for module in flint_host_modules() {
            installed = Some(module.install_from_catalog(registry, catalog)?);
        }
        installed.ok_or_else(|| VmError::HostError("flint host composition is empty".to_string()))
    })
}

use vm::{HostFunctionSchema, HostParamSchema, HostTypeSchema};

/// Compile-visible schemas for residual Torch CONTEXT_HOST_OPS.
///
/// These ops stay args-slice at runtime (`BoundHost` + private
/// `TorchContext`). Schemas live in the same catalog snapshot as the
/// descriptor modules so bundled RSS type-checks against one catalog.
pub fn context_host_op_schemas() -> Vec<HostFunctionSchema> {
    vec![
        HostFunctionSchema::with_return(
            "flint::tokenizer::load",
            vec![HostParamSchema::value("path", HostTypeSchema::String)],
            HostTypeSchema::Bool,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::encode_chat",
            vec![
                HostParamSchema::value("system", HostTypeSchema::String),
                HostParamSchema::value("user", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::encode_vl_chat",
            vec![
                HostParamSchema::value("system", HostTypeSchema::String),
                HostParamSchema::value("user", HostTypeSchema::String),
                HostParamSchema::value("image_tokens", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::encode_padded",
            vec![
                HostParamSchema::value("text", HostTypeSchema::String),
                HostParamSchema::value("max_len", HostTypeSchema::Int),
                HostParamSchema::value("pad_token", HostTypeSchema::Int),
                HostParamSchema::value("add_special_tokens", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::format_token_labels",
            vec![
                HostParamSchema::value("text", HostTypeSchema::String),
                HostParamSchema::value("labels", HostTypeSchema::Int),
                HostParamSchema::value("label_names", HostTypeSchema::String),
                HostParamSchema::value("add_special_tokens", HostTypeSchema::Bool),
            ],
            HostTypeSchema::String,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::decode_generated",
            vec![
                HostParamSchema::value("tokens", HostTypeSchema::Int),
                HostParamSchema::value("prompt_len", HostTypeSchema::Int),
            ],
            HostTypeSchema::String,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::append_token",
            vec![
                HostParamSchema::value("tokens", HostTypeSchema::Int),
                HostParamSchema::value("token", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::append_token_tensor",
            vec![
                HostParamSchema::value("tokens", HostTypeSchema::Int),
                HostParamSchema::value("token", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::clear_generated_tokens",
            vec![],
            HostTypeSchema::Bool,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::push_generated_token_tensor",
            vec![HostParamSchema::value("token", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::decode_generated_tokens",
            vec![],
            HostTypeSchema::String,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::single_token",
            vec![HostParamSchema::value("token", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tokenizer::is_eos",
            vec![HostParamSchema::value("token", HostTypeSchema::Int)],
            HostTypeSchema::Bool,
        ),
        HostFunctionSchema::with_return(
            "flint::weights::load",
            vec![HostParamSchema::value("path", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::weights::get",
            vec![HostParamSchema::value("name", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::weights::get_indexed",
            vec![
                HostParamSchema::value("prefix", HostTypeSchema::String),
                HostParamSchema::value("index", HostTypeSchema::Int),
                HostParamSchema::value("suffix", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::weights::get_or",
            vec![
                HostParamSchema::value("name", HostTypeSchema::String),
                HostParamSchema::value("fallback", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::weights::get_optional",
            vec![HostParamSchema::value("name", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::size",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::shape",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::String,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::save_safetensors",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("path", HostTypeSchema::String),
                HostParamSchema::value("name", HostTypeSchema::String),
            ],
            HostTypeSchema::Bool,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::load_safetensors",
            vec![
                HostParamSchema::value("path", HostTypeSchema::String),
                HostParamSchema::value("name", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::to_float",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::to_bfloat16",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::ones_like",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::zeros_like",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::zeros_like_int",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::arange",
            vec![HostParamSchema::value("end", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::arange_start",
            vec![
                HostParamSchema::value("start", HostTypeSchema::Int),
                HostParamSchema::value("end", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::causal_mask",
            vec![HostParamSchema::value("seq_len", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::causal_padding_mask",
            vec![HostParamSchema::value(
                "attention_mask",
                HostTypeSchema::Int,
            )],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::padding_mask",
            vec![HostParamSchema::value(
                "attention_mask",
                HostTypeSchema::Int,
            )],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::rope_cos",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::rope_sin",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::rope_cos_at",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
                HostParamSchema::value("start", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::rope_sin_at",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
                HostParamSchema::value("start", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::add",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::sub",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::mul",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::add_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::mul_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::div_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::pow_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::mean_dim",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("keepdim", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::rsqrt",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::neg",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::cos",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::sin",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::matmul",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::softmax",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::masked_fill",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("mask", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::cat2",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::stack2",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::chunk",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("chunks", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("index", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::narrow",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("start", HostTypeSchema::Int),
                HostParamSchema::value("len", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::tail",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("len", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::transpose",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim0", HostTypeSchema::Int),
                HostParamSchema::value("dim1", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::unsqueeze",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::repeat_interleave",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("repeats", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::argmax_int",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::argmax",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::argmax_token",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::pad_reflect2d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
                HostParamSchema::value("top", HostTypeSchema::Int),
                HostParamSchema::value("bottom", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::relu",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::sigmoid",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::silu",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::gelu",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("approximate", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::swiglu",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::contiguous",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::permute3",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::permute4",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
                HostParamSchema::value("d3", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::permute5",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
                HostParamSchema::value("d3", HostTypeSchema::Int),
                HostParamSchema::value("d4", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::view2",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::view3",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::view4",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
                HostParamSchema::value("d3", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::view5",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
                HostParamSchema::value("d3", HostTypeSchema::Int),
                HostParamSchema::value("d4", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::select",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("index", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::real",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::imag",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::complex",
            vec![
                HostParamSchema::value("real", HostTypeSchema::Int),
                HostParamSchema::value("imag", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::fft_rfftn2",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::fft_irfftn2",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("height", HostTypeSchema::Int),
                HostParamSchema::value("width", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::tensor::avg_pool2d_2",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::embedding",
            vec![
                HostParamSchema::value("indices", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::linear",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("bias", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::layer_norm",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("bias", HostTypeSchema::Int),
                HostParamSchema::value("eps", HostTypeSchema::Float),
                HostParamSchema::value("normalized_size", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::swiglu_linear",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("bias", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::rms_norm",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("eps", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::add_rms_norm",
            vec![
                HostParamSchema::value("input", HostTypeSchema::Int),
                HostParamSchema::value("residual", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("eps", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::apply_rope",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("cos", HostTypeSchema::Int),
                HostParamSchema::value("sin", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::apply_rope_pair",
            vec![
                HostParamSchema::value("query", HostTypeSchema::Int),
                HostParamSchema::value("key", HostTypeSchema::Int),
                HostParamSchema::value("cos", HostTypeSchema::Int),
                HostParamSchema::value("sin", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::scaled_dot_product_attention",
            vec![
                HostParamSchema::value("query", HostTypeSchema::Int),
                HostParamSchema::value("key", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Int),
                HostParamSchema::value("is_causal", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::scaled_dot_product_attention_masked",
            vec![
                HostParamSchema::value("query", HostTypeSchema::Int),
                HostParamSchema::value("key", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Int),
                HostParamSchema::value("mask", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::conv1d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("bias", HostTypeSchema::Int),
                HostParamSchema::value("stride", HostTypeSchema::Int),
                HostParamSchema::value("padding", HostTypeSchema::Int),
                HostParamSchema::value("dilation", HostTypeSchema::Int),
                HostParamSchema::value("groups", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::conv1d_step",
            vec![
                HostParamSchema::value("state", HostTypeSchema::Int),
                HostParamSchema::value("input", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::conv2d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight_prefix", HostTypeSchema::String),
                HostParamSchema::value("stride", HostTypeSchema::Int),
                HostParamSchema::value("padding", HostTypeSchema::Int),
                HostParamSchema::value("reflect", HostTypeSchema::Bool),
                HostParamSchema::value("bias", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::conv_transpose2d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight_prefix", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::nn::batch_norm2d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight_prefix", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::image::lfm2_vl_patches",
            vec![HostParamSchema::value("path", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::vl::siglip2_position_embedding",
            vec![
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("height", HostTypeSchema::Int),
                HostParamSchema::value("width", HostTypeSchema::Int),
                HostParamSchema::value("max_len", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::vl::pixel_unshuffle2",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        HostFunctionSchema::with_return(
            "flint::vl::scatter_image_embeddings",
            vec![
                HostParamSchema::value("input_ids", HostTypeSchema::Int),
                HostParamSchema::value("inputs_embeds", HostTypeSchema::Int),
                HostParamSchema::value("image_features", HostTypeSchema::Int),
                HostParamSchema::value("image_token_id", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
    ]
}

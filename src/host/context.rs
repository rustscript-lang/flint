use vm::{
    CallOutcome, HostAdapterDescriptor, HostBindingDescriptor, HostBindingKind,
    HostFunctionDescriptor, HostFunctionSchema, HostModuleDescriptor, HostParamSchema,
    HostTypeSchema, Value, VmResult,
};

use super::{TorchContext, with_context};

fn call_context_host(
    name: &'static str,
    op: fn(&mut TorchContext, &[Value]) -> VmResult<CallOutcome>,
    args: &[Value],
) -> VmResult<CallOutcome> {
    with_context(|context| {
        let previous = context.active_host_op.replace(name);
        let outcome = if context.host_op_profile_enabled {
            let started = std::time::Instant::now();
            let outcome = op(context, args);
            context.record_host_op(name, started.elapsed());
            outcome
        } else {
            op(context, args)
        };
        context.active_host_op = previous;
        outcome
    })
}

fn tokenizer_load_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tokenizer::load", super::tokenizer_load, args)
}

pub(super) fn tokenizer_load_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::load",
            vec![HostParamSchema::value("path", HostTypeSchema::String)],
            HostTypeSchema::Bool,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_load_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_encode_chat_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::encode_chat",
        super::tokenizer_encode_chat,
        args,
    )
}

pub(super) fn tokenizer_encode_chat_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::encode_chat",
            vec![
                HostParamSchema::value("system", HostTypeSchema::String),
                HostParamSchema::value("user", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_encode_chat_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_encode_vl_chat_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::encode_vl_chat",
        super::tokenizer_encode_vl_chat,
        args,
    )
}

pub(super) fn tokenizer_encode_vl_chat_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::encode_vl_chat",
            vec![
                HostParamSchema::value("system", HostTypeSchema::String),
                HostParamSchema::value("user", HostTypeSchema::String),
                HostParamSchema::value("image_tokens", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_encode_vl_chat_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_encode_padded_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::encode_padded",
        super::tokenizer_encode_padded,
        args,
    )
}

pub(super) fn tokenizer_encode_padded_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::encode_padded",
            vec![
                HostParamSchema::value("text", HostTypeSchema::String),
                HostParamSchema::value("max_len", HostTypeSchema::Int),
                HostParamSchema::value("pad_token", HostTypeSchema::Int),
                HostParamSchema::value("add_special_tokens", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_encode_padded_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_format_token_labels_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::format_token_labels",
        super::tokenizer_format_token_labels,
        args,
    )
}

pub(super) fn tokenizer_format_token_labels_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::format_token_labels",
            vec![
                HostParamSchema::value("text", HostTypeSchema::String),
                HostParamSchema::value("labels", HostTypeSchema::Int),
                HostParamSchema::value("label_names", HostTypeSchema::String),
                HostParamSchema::value("add_special_tokens", HostTypeSchema::Bool),
            ],
            HostTypeSchema::String,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_format_token_labels_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_decode_generated_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::decode_generated",
        super::tokenizer_decode_generated,
        args,
    )
}

pub(super) fn tokenizer_decode_generated_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::decode_generated",
            vec![
                HostParamSchema::value("tokens", HostTypeSchema::Int),
                HostParamSchema::value("prompt_len", HostTypeSchema::Int),
            ],
            HostTypeSchema::String,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_decode_generated_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_append_token_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::append_token",
        super::tokenizer_append_token,
        args,
    )
}

pub(super) fn tokenizer_append_token_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::append_token",
            vec![
                HostParamSchema::value("tokens", HostTypeSchema::Int),
                HostParamSchema::value("token", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_append_token_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_append_token_tensor_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::append_token_tensor",
        super::tokenizer_append_token_tensor,
        args,
    )
}

pub(super) fn tokenizer_append_token_tensor_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::append_token_tensor",
            vec![
                HostParamSchema::value("tokens", HostTypeSchema::Int),
                HostParamSchema::value("token", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_append_token_tensor_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_clear_generated_tokens_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::clear_generated_tokens",
        super::tokenizer_clear_generated_tokens,
        args,
    )
}

pub(super) fn tokenizer_clear_generated_tokens_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::clear_generated_tokens",
            vec![],
            HostTypeSchema::Bool,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_clear_generated_tokens_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_push_generated_token_tensor_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::push_generated_token_tensor",
        super::tokenizer_push_generated_token_tensor,
        args,
    )
}

pub(super) fn tokenizer_push_generated_token_tensor_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::push_generated_token_tensor",
            vec![HostParamSchema::value("token", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_push_generated_token_tensor_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_decode_generated_tokens_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::decode_generated_tokens",
        super::tokenizer_decode_generated_tokens,
        args,
    )
}

pub(super) fn tokenizer_decode_generated_tokens_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::decode_generated_tokens",
            vec![],
            HostTypeSchema::String,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_decode_generated_tokens_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_single_token_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tokenizer::single_token",
        super::tokenizer_single_token,
        args,
    )
}

pub(super) fn tokenizer_single_token_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::single_token",
            vec![HostParamSchema::value("token", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_single_token_args),
        resource_types: Vec::new(),
    }
}

fn tokenizer_is_eos_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tokenizer::is_eos", super::tokenizer_is_eos, args)
}

pub(super) fn tokenizer_is_eos_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tokenizer::is_eos",
            vec![HostParamSchema::value("token", HostTypeSchema::Int)],
            HostTypeSchema::Bool,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tokenizer_is_eos_args),
        resource_types: Vec::new(),
    }
}

fn weights_load_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::weights::load", super::weights_load, args)
}

pub(super) fn weights_load_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::weights::load",
            vec![HostParamSchema::value("path", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(weights_load_args),
        resource_types: Vec::new(),
    }
}

fn weights_get_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::weights::get", super::weights_get, args)
}

pub(super) fn weights_get_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::weights::get",
            vec![HostParamSchema::value("name", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(weights_get_args),
        resource_types: Vec::new(),
    }
}

fn weights_get_indexed_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::weights::get_indexed",
        super::weights_get_indexed,
        args,
    )
}

pub(super) fn weights_get_indexed_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::weights::get_indexed",
            vec![
                HostParamSchema::value("prefix", HostTypeSchema::String),
                HostParamSchema::value("index", HostTypeSchema::Int),
                HostParamSchema::value("suffix", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(weights_get_indexed_args),
        resource_types: Vec::new(),
    }
}

fn weights_get_or_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::weights::get_or", super::weights_get_or, args)
}

pub(super) fn weights_get_or_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::weights::get_or",
            vec![
                HostParamSchema::value("name", HostTypeSchema::String),
                HostParamSchema::value("fallback", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(weights_get_or_args),
        resource_types: Vec::new(),
    }
}

fn weights_get_optional_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::weights::get_optional",
        super::weights_get_optional,
        args,
    )
}

pub(super) fn weights_get_optional_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::weights::get_optional",
            vec![HostParamSchema::value("name", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(weights_get_optional_args),
        resource_types: Vec::new(),
    }
}

fn tensor_size_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::size", super::tensor_size, args)
}

pub(super) fn tensor_size_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::size",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_size_args),
        resource_types: Vec::new(),
    }
}

fn tensor_shape_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::shape", super::tensor_shape, args)
}

pub(super) fn tensor_shape_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::shape",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::String,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_shape_args),
        resource_types: Vec::new(),
    }
}

fn tensor_save_safetensors_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::save_safetensors",
        super::tensor_save_safetensors,
        args,
    )
}

pub(super) fn tensor_save_safetensors_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::save_safetensors",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("path", HostTypeSchema::String),
                HostParamSchema::value("name", HostTypeSchema::String),
            ],
            HostTypeSchema::Bool,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_save_safetensors_args),
        resource_types: Vec::new(),
    }
}

fn tensor_load_safetensors_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::load_safetensors",
        super::tensor_load_safetensors,
        args,
    )
}

pub(super) fn tensor_load_safetensors_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::load_safetensors",
            vec![
                HostParamSchema::value("path", HostTypeSchema::String),
                HostParamSchema::value("name", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_load_safetensors_args),
        resource_types: Vec::new(),
    }
}

fn tensor_to_float_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::to_float", super::tensor_to_float, args)
}

pub(super) fn tensor_to_float_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::to_float",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_to_float_args),
        resource_types: Vec::new(),
    }
}

fn tensor_to_bfloat16_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::to_bfloat16",
        super::tensor_to_bfloat16,
        args,
    )
}

pub(super) fn tensor_to_bfloat16_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::to_bfloat16",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_to_bfloat16_args),
        resource_types: Vec::new(),
    }
}

fn tensor_ones_like_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::ones_like", super::tensor_ones_like, args)
}

pub(super) fn tensor_ones_like_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::ones_like",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_ones_like_args),
        resource_types: Vec::new(),
    }
}

fn tensor_zeros_like_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::zeros_like", super::tensor_zeros_like, args)
}

pub(super) fn tensor_zeros_like_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::zeros_like",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_zeros_like_args),
        resource_types: Vec::new(),
    }
}

fn tensor_zeros_like_int_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::zeros_like_int",
        super::tensor_zeros_like_int,
        args,
    )
}

pub(super) fn tensor_zeros_like_int_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::zeros_like_int",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_zeros_like_int_args),
        resource_types: Vec::new(),
    }
}

fn tensor_arange_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::arange", super::tensor_arange, args)
}

pub(super) fn tensor_arange_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::arange",
            vec![HostParamSchema::value("end", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_arange_args),
        resource_types: Vec::new(),
    }
}

fn tensor_arange_start_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::arange_start",
        super::tensor_arange_start,
        args,
    )
}

pub(super) fn tensor_arange_start_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::arange_start",
            vec![
                HostParamSchema::value("start", HostTypeSchema::Int),
                HostParamSchema::value("end", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_arange_start_args),
        resource_types: Vec::new(),
    }
}

fn tensor_causal_mask_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::causal_mask",
        super::tensor_causal_mask,
        args,
    )
}

pub(super) fn tensor_causal_mask_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::causal_mask",
            vec![HostParamSchema::value("seq_len", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_causal_mask_args),
        resource_types: Vec::new(),
    }
}

fn tensor_causal_padding_mask_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::causal_padding_mask",
        super::tensor_causal_padding_mask,
        args,
    )
}

pub(super) fn tensor_causal_padding_mask_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::causal_padding_mask",
            vec![HostParamSchema::value(
                "attention_mask",
                HostTypeSchema::Int,
            )],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_causal_padding_mask_args),
        resource_types: Vec::new(),
    }
}

fn tensor_padding_mask_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::padding_mask",
        super::tensor_padding_mask,
        args,
    )
}

pub(super) fn tensor_padding_mask_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::padding_mask",
            vec![HostParamSchema::value(
                "attention_mask",
                HostTypeSchema::Int,
            )],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_padding_mask_args),
        resource_types: Vec::new(),
    }
}

fn tensor_rope_cos_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::rope_cos", super::tensor_rope_cos, args)
}

pub(super) fn tensor_rope_cos_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::rope_cos",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_rope_cos_args),
        resource_types: Vec::new(),
    }
}

fn tensor_rope_sin_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::rope_sin", super::tensor_rope_sin, args)
}

pub(super) fn tensor_rope_sin_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::rope_sin",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_rope_sin_args),
        resource_types: Vec::new(),
    }
}

fn tensor_rope_cos_at_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::rope_cos_at",
        super::tensor_rope_cos_at,
        args,
    )
}

pub(super) fn tensor_rope_cos_at_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::rope_cos_at",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
                HostParamSchema::value("start", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_rope_cos_at_args),
        resource_types: Vec::new(),
    }
}

fn tensor_rope_sin_at_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::rope_sin_at",
        super::tensor_rope_sin_at,
        args,
    )
}

pub(super) fn tensor_rope_sin_at_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::rope_sin_at",
            vec![
                HostParamSchema::value("seq_len", HostTypeSchema::Int),
                HostParamSchema::value("head_dim", HostTypeSchema::Int),
                HostParamSchema::value("theta", HostTypeSchema::Float),
                HostParamSchema::value("start", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_rope_sin_at_args),
        resource_types: Vec::new(),
    }
}

fn tensor_add_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::add", super::tensor_add, args)
}

pub(super) fn tensor_add_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::add",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_add_args),
        resource_types: Vec::new(),
    }
}

fn tensor_sub_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::sub", super::tensor_sub, args)
}

pub(super) fn tensor_sub_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::sub",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_sub_args),
        resource_types: Vec::new(),
    }
}

fn tensor_mul_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::mul", super::tensor_mul, args)
}

pub(super) fn tensor_mul_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::mul",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_mul_args),
        resource_types: Vec::new(),
    }
}

fn tensor_add_scalar_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::add_scalar", super::tensor_add_scalar, args)
}

pub(super) fn tensor_add_scalar_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::add_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_add_scalar_args),
        resource_types: Vec::new(),
    }
}

fn tensor_mul_scalar_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::mul_scalar", super::tensor_mul_scalar, args)
}

pub(super) fn tensor_mul_scalar_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::mul_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_mul_scalar_args),
        resource_types: Vec::new(),
    }
}

fn tensor_div_scalar_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::div_scalar", super::tensor_div_scalar, args)
}

pub(super) fn tensor_div_scalar_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::div_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_div_scalar_args),
        resource_types: Vec::new(),
    }
}

fn tensor_pow_scalar_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::pow_scalar", super::tensor_pow_scalar, args)
}

pub(super) fn tensor_pow_scalar_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::pow_scalar",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_pow_scalar_args),
        resource_types: Vec::new(),
    }
}

fn tensor_mean_dim_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::mean_dim", super::tensor_mean_dim, args)
}

pub(super) fn tensor_mean_dim_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::mean_dim",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("keepdim", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_mean_dim_args),
        resource_types: Vec::new(),
    }
}

fn tensor_rsqrt_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::rsqrt", super::tensor_rsqrt, args)
}

pub(super) fn tensor_rsqrt_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::rsqrt",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_rsqrt_args),
        resource_types: Vec::new(),
    }
}

fn tensor_neg_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::neg", super::tensor_neg, args)
}

pub(super) fn tensor_neg_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::neg",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_neg_args),
        resource_types: Vec::new(),
    }
}

fn tensor_cos_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::cos", super::tensor_cos, args)
}

pub(super) fn tensor_cos_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::cos",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_cos_args),
        resource_types: Vec::new(),
    }
}

fn tensor_sin_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::sin", super::tensor_sin, args)
}

pub(super) fn tensor_sin_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::sin",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_sin_args),
        resource_types: Vec::new(),
    }
}

fn tensor_matmul_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::matmul", super::tensor_matmul, args)
}

pub(super) fn tensor_matmul_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::matmul",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_matmul_args),
        resource_types: Vec::new(),
    }
}

fn tensor_softmax_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::softmax", super::tensor_softmax, args)
}

pub(super) fn tensor_softmax_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::softmax",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_softmax_args),
        resource_types: Vec::new(),
    }
}

fn tensor_masked_fill_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::masked_fill",
        super::tensor_masked_fill,
        args,
    )
}

pub(super) fn tensor_masked_fill_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::masked_fill",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("mask", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_masked_fill_args),
        resource_types: Vec::new(),
    }
}

fn tensor_cat2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::cat2", super::tensor_cat2, args)
}

pub(super) fn tensor_cat2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::cat2",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_cat2_args),
        resource_types: Vec::new(),
    }
}

fn tensor_stack2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::stack2", super::tensor_stack2, args)
}

pub(super) fn tensor_stack2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::stack2",
            vec![
                HostParamSchema::value("left", HostTypeSchema::Int),
                HostParamSchema::value("right", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_stack2_args),
        resource_types: Vec::new(),
    }
}

fn tensor_chunk_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::chunk", super::tensor_chunk, args)
}

pub(super) fn tensor_chunk_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::chunk",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("chunks", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("index", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_chunk_args),
        resource_types: Vec::new(),
    }
}

fn tensor_narrow_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::narrow", super::tensor_narrow, args)
}

pub(super) fn tensor_narrow_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::narrow",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("start", HostTypeSchema::Int),
                HostParamSchema::value("len", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_narrow_args),
        resource_types: Vec::new(),
    }
}

fn tensor_tail_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::tail", super::tensor_tail, args)
}

pub(super) fn tensor_tail_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::tail",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("len", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_tail_args),
        resource_types: Vec::new(),
    }
}

fn tensor_transpose_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::transpose", super::tensor_transpose, args)
}

pub(super) fn tensor_transpose_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::transpose",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim0", HostTypeSchema::Int),
                HostParamSchema::value("dim1", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_transpose_args),
        resource_types: Vec::new(),
    }
}

fn tensor_unsqueeze_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::unsqueeze", super::tensor_unsqueeze, args)
}

pub(super) fn tensor_unsqueeze_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::unsqueeze",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_unsqueeze_args),
        resource_types: Vec::new(),
    }
}

fn tensor_repeat_interleave_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::repeat_interleave",
        super::tensor_repeat_interleave,
        args,
    )
}

pub(super) fn tensor_repeat_interleave_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::repeat_interleave",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("repeats", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_repeat_interleave_args),
        resource_types: Vec::new(),
    }
}

fn tensor_argmax_int_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::argmax_int", super::tensor_argmax_int, args)
}

pub(super) fn tensor_argmax_int_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::argmax_int",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_argmax_int_args),
        resource_types: Vec::new(),
    }
}

fn tensor_argmax_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::argmax", super::tensor_argmax, args)
}

pub(super) fn tensor_argmax_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::argmax",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_argmax_args),
        resource_types: Vec::new(),
    }
}

fn tensor_argmax_token_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::argmax_token",
        super::tensor_argmax_token,
        args,
    )
}

pub(super) fn tensor_argmax_token_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::argmax_token",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_argmax_token_args),
        resource_types: Vec::new(),
    }
}

fn tensor_pad_reflect2d_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::pad_reflect2d",
        super::tensor_pad_reflect2d,
        args,
    )
}

pub(super) fn tensor_pad_reflect2d_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_pad_reflect2d_args),
        resource_types: Vec::new(),
    }
}

fn tensor_relu_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::relu", super::tensor_relu, args)
}

pub(super) fn tensor_relu_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::relu",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_relu_args),
        resource_types: Vec::new(),
    }
}

fn tensor_sigmoid_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::sigmoid", super::tensor_sigmoid, args)
}

pub(super) fn tensor_sigmoid_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::sigmoid",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_sigmoid_args),
        resource_types: Vec::new(),
    }
}

fn tensor_silu_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::silu", super::tensor_silu, args)
}

pub(super) fn tensor_silu_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::silu",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_silu_args),
        resource_types: Vec::new(),
    }
}

fn tensor_gelu_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::gelu", super::tensor_gelu, args)
}

pub(super) fn tensor_gelu_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::gelu",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("approximate", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_gelu_args),
        resource_types: Vec::new(),
    }
}

fn tensor_swiglu_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::swiglu", super::tensor_swiglu, args)
}

pub(super) fn tensor_swiglu_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::swiglu",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_swiglu_args),
        resource_types: Vec::new(),
    }
}

fn tensor_contiguous_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::contiguous", super::tensor_contiguous, args)
}

pub(super) fn tensor_contiguous_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::contiguous",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_contiguous_args),
        resource_types: Vec::new(),
    }
}

fn tensor_permute3_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::permute3", super::tensor_permute3, args)
}

pub(super) fn tensor_permute3_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::permute3",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_permute3_args),
        resource_types: Vec::new(),
    }
}

fn tensor_permute4_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::permute4", super::tensor_permute4, args)
}

pub(super) fn tensor_permute4_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_permute4_args),
        resource_types: Vec::new(),
    }
}

fn tensor_permute5_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::permute5", super::tensor_permute5, args)
}

pub(super) fn tensor_permute5_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_permute5_args),
        resource_types: Vec::new(),
    }
}

fn tensor_view2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::view2", super::tensor_view2, args)
}

pub(super) fn tensor_view2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::view2",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_view2_args),
        resource_types: Vec::new(),
    }
}

fn tensor_view3_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::view3", super::tensor_view3, args)
}

pub(super) fn tensor_view3_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::view3",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("d0", HostTypeSchema::Int),
                HostParamSchema::value("d1", HostTypeSchema::Int),
                HostParamSchema::value("d2", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_view3_args),
        resource_types: Vec::new(),
    }
}

fn tensor_view4_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::view4", super::tensor_view4, args)
}

pub(super) fn tensor_view4_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_view4_args),
        resource_types: Vec::new(),
    }
}

fn tensor_view5_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::view5", super::tensor_view5, args)
}

pub(super) fn tensor_view5_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_view5_args),
        resource_types: Vec::new(),
    }
}

fn tensor_select_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::select", super::tensor_select, args)
}

pub(super) fn tensor_select_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::select",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("dim", HostTypeSchema::Int),
                HostParamSchema::value("index", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_select_args),
        resource_types: Vec::new(),
    }
}

fn tensor_real_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::real", super::tensor_real, args)
}

pub(super) fn tensor_real_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::real",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_real_args),
        resource_types: Vec::new(),
    }
}

fn tensor_imag_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::imag", super::tensor_imag, args)
}

pub(super) fn tensor_imag_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::imag",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_imag_args),
        resource_types: Vec::new(),
    }
}

fn tensor_complex_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::complex", super::tensor_complex, args)
}

pub(super) fn tensor_complex_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::complex",
            vec![
                HostParamSchema::value("real", HostTypeSchema::Int),
                HostParamSchema::value("imag", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_complex_args),
        resource_types: Vec::new(),
    }
}

fn tensor_fft_rfftn2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::tensor::fft_rfftn2", super::tensor_fft_rfftn2, args)
}

pub(super) fn tensor_fft_rfftn2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::fft_rfftn2",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_fft_rfftn2_args),
        resource_types: Vec::new(),
    }
}

fn tensor_fft_irfftn2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::fft_irfftn2",
        super::tensor_fft_irfftn2,
        args,
    )
}

pub(super) fn tensor_fft_irfftn2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::fft_irfftn2",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("height", HostTypeSchema::Int),
                HostParamSchema::value("width", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_fft_irfftn2_args),
        resource_types: Vec::new(),
    }
}

fn tensor_avg_pool2d_2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::tensor::avg_pool2d_2",
        super::tensor_avg_pool2d_2,
        args,
    )
}

pub(super) fn tensor_avg_pool2d_2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::tensor::avg_pool2d_2",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(tensor_avg_pool2d_2_args),
        resource_types: Vec::new(),
    }
}

fn nn_embedding_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::embedding", super::nn_embedding, args)
}

pub(super) fn nn_embedding_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::embedding",
            vec![
                HostParamSchema::value("indices", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_embedding_args),
        resource_types: Vec::new(),
    }
}

fn nn_linear_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::linear", super::nn_linear, args)
}

pub(super) fn nn_linear_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::linear",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("bias", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_linear_args),
        resource_types: Vec::new(),
    }
}

fn nn_layer_norm_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::layer_norm", super::nn_layer_norm, args)
}

pub(super) fn nn_layer_norm_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_layer_norm_args),
        resource_types: Vec::new(),
    }
}

fn nn_swiglu_linear_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::swiglu_linear", super::nn_swiglu_linear, args)
}

pub(super) fn nn_swiglu_linear_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::swiglu_linear",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("bias", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_swiglu_linear_args),
        resource_types: Vec::new(),
    }
}

fn nn_rms_norm_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::rms_norm", super::nn_rms_norm, args)
}

pub(super) fn nn_rms_norm_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::rms_norm",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("eps", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_rms_norm_args),
        resource_types: Vec::new(),
    }
}

fn nn_add_rms_norm_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::add_rms_norm", super::nn_add_rms_norm, args)
}

pub(super) fn nn_add_rms_norm_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::add_rms_norm",
            vec![
                HostParamSchema::value("input", HostTypeSchema::Int),
                HostParamSchema::value("residual", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("eps", HostTypeSchema::Float),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_add_rms_norm_args),
        resource_types: Vec::new(),
    }
}

fn nn_apply_rope_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::apply_rope", super::nn_apply_rope, args)
}

pub(super) fn nn_apply_rope_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::apply_rope",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("cos", HostTypeSchema::Int),
                HostParamSchema::value("sin", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_apply_rope_args),
        resource_types: Vec::new(),
    }
}

fn nn_apply_rope_pair_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::nn::apply_rope_pair",
        super::nn_apply_rope_pair,
        args,
    )
}

pub(super) fn nn_apply_rope_pair_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::apply_rope_pair",
            vec![
                HostParamSchema::value("query", HostTypeSchema::Int),
                HostParamSchema::value("key", HostTypeSchema::Int),
                HostParamSchema::value("cos", HostTypeSchema::Int),
                HostParamSchema::value("sin", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_apply_rope_pair_args),
        resource_types: Vec::new(),
    }
}

fn nn_scaled_dot_product_attention_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::nn::scaled_dot_product_attention",
        super::nn_scaled_dot_product_attention,
        args,
    )
}

pub(super) fn nn_scaled_dot_product_attention_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::scaled_dot_product_attention",
            vec![
                HostParamSchema::value("query", HostTypeSchema::Int),
                HostParamSchema::value("key", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Int),
                HostParamSchema::value("is_causal", HostTypeSchema::Bool),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_scaled_dot_product_attention_args),
        resource_types: Vec::new(),
    }
}

fn nn_scaled_dot_product_attention_masked_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::nn::scaled_dot_product_attention_masked",
        super::nn_scaled_dot_product_attention_masked,
        args,
    )
}

pub(super) fn nn_scaled_dot_product_attention_masked_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::scaled_dot_product_attention_masked",
            vec![
                HostParamSchema::value("query", HostTypeSchema::Int),
                HostParamSchema::value("key", HostTypeSchema::Int),
                HostParamSchema::value("value", HostTypeSchema::Int),
                HostParamSchema::value("mask", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_scaled_dot_product_attention_masked_args),
        resource_types: Vec::new(),
    }
}

fn nn_conv1d_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::conv1d", super::nn_conv1d, args)
}

pub(super) fn nn_conv1d_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_conv1d_args),
        resource_types: Vec::new(),
    }
}

fn nn_conv1d_step_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::conv1d_step", super::nn_conv1d_step, args)
}

pub(super) fn nn_conv1d_step_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::conv1d_step",
            vec![
                HostParamSchema::value("state", HostTypeSchema::Int),
                HostParamSchema::value("input", HostTypeSchema::Int),
                HostParamSchema::value("weight", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_conv1d_step_args),
        resource_types: Vec::new(),
    }
}

fn nn_conv2d_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::conv2d", super::nn_conv2d, args)
}

pub(super) fn nn_conv2d_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
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
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_conv2d_args),
        resource_types: Vec::new(),
    }
}

fn nn_conv_transpose2d_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::nn::conv_transpose2d",
        super::nn_conv_transpose2d,
        args,
    )
}

pub(super) fn nn_conv_transpose2d_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::conv_transpose2d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight_prefix", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_conv_transpose2d_args),
        resource_types: Vec::new(),
    }
}

fn nn_batch_norm2d_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host("flint::nn::batch_norm2d", super::nn_batch_norm2d, args)
}

pub(super) fn nn_batch_norm2d_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::nn::batch_norm2d",
            vec![
                HostParamSchema::value("tensor", HostTypeSchema::Int),
                HostParamSchema::value("weight_prefix", HostTypeSchema::String),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(nn_batch_norm2d_args),
        resource_types: Vec::new(),
    }
}

fn image_lfm2_vl_patches_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::image::lfm2_vl_patches",
        super::image_lfm2_vl_patches,
        args,
    )
}

pub(super) fn image_lfm2_vl_patches_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::image::lfm2_vl_patches",
            vec![HostParamSchema::value("path", HostTypeSchema::String)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(image_lfm2_vl_patches_args),
        resource_types: Vec::new(),
    }
}

fn vl_siglip2_position_embedding_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::vl::siglip2_position_embedding",
        super::vl_siglip2_position_embedding,
        args,
    )
}

pub(super) fn vl_siglip2_position_embedding_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::vl::siglip2_position_embedding",
            vec![
                HostParamSchema::value("weight", HostTypeSchema::Int),
                HostParamSchema::value("height", HostTypeSchema::Int),
                HostParamSchema::value("width", HostTypeSchema::Int),
                HostParamSchema::value("max_len", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(vl_siglip2_position_embedding_args),
        resource_types: Vec::new(),
    }
}

fn vl_pixel_unshuffle2_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::vl::pixel_unshuffle2",
        super::vl_pixel_unshuffle2,
        args,
    )
}

pub(super) fn vl_pixel_unshuffle2_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::vl::pixel_unshuffle2",
            vec![HostParamSchema::value("tensor", HostTypeSchema::Int)],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(vl_pixel_unshuffle2_args),
        resource_types: Vec::new(),
    }
}

fn vl_scatter_image_embeddings_args(args: &[Value]) -> VmResult<CallOutcome> {
    call_context_host(
        "flint::vl::scatter_image_embeddings",
        super::vl_scatter_image_embeddings,
        args,
    )
}

pub(super) fn vl_scatter_image_embeddings_descriptor() -> HostFunctionDescriptor {
    HostFunctionDescriptor {
        schema: HostFunctionSchema::with_return(
            "flint::vl::scatter_image_embeddings",
            vec![
                HostParamSchema::value("input_ids", HostTypeSchema::Int),
                HostParamSchema::value("inputs_embeds", HostTypeSchema::Int),
                HostParamSchema::value("image_features", HostTypeSchema::Int),
                HostParamSchema::value("image_token_id", HostTypeSchema::Int),
            ],
            HostTypeSchema::Int,
        ),
        binding: HostBindingDescriptor {
            kind: HostBindingKind::StaticArgs,
        },
        effects: Vec::new(),
        adapter: HostAdapterDescriptor::StaticArgs(vl_scatter_image_embeddings_args),
        resource_types: Vec::new(),
    }
}

pub(super) fn tokenizer_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.tokenizer",
        functions: &[
            tokenizer_load_descriptor,
            tokenizer_encode_chat_descriptor,
            tokenizer_encode_vl_chat_descriptor,
            tokenizer_encode_padded_descriptor,
            tokenizer_format_token_labels_descriptor,
            tokenizer_decode_generated_descriptor,
            tokenizer_append_token_descriptor,
            tokenizer_append_token_tensor_descriptor,
            tokenizer_clear_generated_tokens_descriptor,
            tokenizer_push_generated_token_tensor_descriptor,
            tokenizer_decode_generated_tokens_descriptor,
            tokenizer_single_token_descriptor,
            tokenizer_is_eos_descriptor,
        ],
        resources: &[],
    }
}

pub(super) fn weights_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.weights",
        functions: &[
            weights_load_descriptor,
            weights_get_descriptor,
            weights_get_indexed_descriptor,
            weights_get_or_descriptor,
            weights_get_optional_descriptor,
        ],
        resources: &[],
    }
}

pub(super) fn tensor_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.tensor",
        functions: &[
            tensor_size_descriptor,
            tensor_shape_descriptor,
            tensor_save_safetensors_descriptor,
            tensor_load_safetensors_descriptor,
            tensor_to_float_descriptor,
            tensor_to_bfloat16_descriptor,
            tensor_ones_like_descriptor,
            tensor_zeros_like_descriptor,
            tensor_zeros_like_int_descriptor,
            tensor_arange_descriptor,
            tensor_arange_start_descriptor,
            tensor_causal_mask_descriptor,
            tensor_causal_padding_mask_descriptor,
            tensor_padding_mask_descriptor,
            tensor_rope_cos_descriptor,
            tensor_rope_sin_descriptor,
            tensor_rope_cos_at_descriptor,
            tensor_rope_sin_at_descriptor,
            tensor_add_descriptor,
            tensor_sub_descriptor,
            tensor_mul_descriptor,
            tensor_add_scalar_descriptor,
            tensor_mul_scalar_descriptor,
            tensor_div_scalar_descriptor,
            tensor_pow_scalar_descriptor,
            tensor_mean_dim_descriptor,
            tensor_rsqrt_descriptor,
            tensor_neg_descriptor,
            tensor_cos_descriptor,
            tensor_sin_descriptor,
            tensor_matmul_descriptor,
            tensor_softmax_descriptor,
            tensor_masked_fill_descriptor,
            tensor_cat2_descriptor,
            tensor_stack2_descriptor,
            tensor_chunk_descriptor,
            tensor_narrow_descriptor,
            tensor_tail_descriptor,
            tensor_transpose_descriptor,
            tensor_unsqueeze_descriptor,
            tensor_repeat_interleave_descriptor,
            tensor_argmax_int_descriptor,
            tensor_argmax_descriptor,
            tensor_argmax_token_descriptor,
            tensor_pad_reflect2d_descriptor,
            tensor_relu_descriptor,
            tensor_sigmoid_descriptor,
            tensor_silu_descriptor,
            tensor_gelu_descriptor,
            tensor_swiglu_descriptor,
            tensor_contiguous_descriptor,
            tensor_permute3_descriptor,
            tensor_permute4_descriptor,
            tensor_permute5_descriptor,
            tensor_view2_descriptor,
            tensor_view3_descriptor,
            tensor_view4_descriptor,
            tensor_view5_descriptor,
            tensor_select_descriptor,
            tensor_real_descriptor,
            tensor_imag_descriptor,
            tensor_complex_descriptor,
            tensor_fft_rfftn2_descriptor,
            tensor_fft_irfftn2_descriptor,
            tensor_avg_pool2d_2_descriptor,
        ],
        resources: &[],
    }
}

pub(super) fn nn_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.nn",
        functions: &[
            nn_embedding_descriptor,
            nn_linear_descriptor,
            nn_layer_norm_descriptor,
            nn_swiglu_linear_descriptor,
            nn_rms_norm_descriptor,
            nn_add_rms_norm_descriptor,
            nn_apply_rope_descriptor,
            nn_apply_rope_pair_descriptor,
            nn_scaled_dot_product_attention_descriptor,
            nn_scaled_dot_product_attention_masked_descriptor,
            nn_conv1d_descriptor,
            nn_conv1d_step_descriptor,
            nn_conv2d_descriptor,
            nn_conv_transpose2d_descriptor,
            nn_batch_norm2d_descriptor,
        ],
        resources: &[],
    }
}

pub(super) fn image_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.image",
        functions: &[image_lfm2_vl_patches_descriptor],
        resources: &[],
    }
}

pub(super) fn vl_host_module() -> HostModuleDescriptor {
    HostModuleDescriptor {
        name: "flint.vl",
        functions: &[
            vl_siglip2_position_embedding_descriptor,
            vl_pixel_unshuffle2_descriptor,
            vl_scatter_image_embeddings_descriptor,
        ],
        resources: &[],
    }
}

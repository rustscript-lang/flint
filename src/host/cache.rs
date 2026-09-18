use pd_host_function::pd_host_function;

use crate::VmResult;

use super::{host_error, with_context};

/// Clears named tensors cached during the current script execution.
#[pd_host_function(name = "flint::cache::clear")]
pub(super) fn cache_clear_impl() -> VmResult<bool> {
    with_context(|context| {
        context.cache.clear();
        Ok(true)
    })
}

/// Returns whether a named cached tensor exists.
#[pd_host_function(name = "flint::cache::has")]
pub(super) fn cache_has_impl(name: &str) -> VmResult<bool> {
    with_context(|context| Ok(context.cache.contains_key(name)))
}

/// Returns a named cached tensor handle.
#[pd_host_function(name = "flint::cache::get")]
pub(super) fn cache_get_impl(name: &str) -> VmResult<i64> {
    with_context(|context| {
        let tensor = context
            .cache
            .get(name)
            .ok_or_else(|| host_error(format!("missing cache tensor '{name}'")))?
            .shallow_clone();
        Ok(context.insert_tensor(tensor))
    })
}

/// Stores a tensor handle in the current execution cache.
#[pd_host_function(name = "flint::cache::set")]
pub(super) fn cache_set_impl(name: &str, tensor: i64) -> VmResult<bool> {
    with_context(|context| {
        let tensor = context.tensor(tensor)?.shallow_clone();
        context.cache.insert(name.to_owned(), tensor);
        Ok(true)
    })
}

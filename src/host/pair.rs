use pd_host_function::pd_host_function;

use crate::VmResult;

use super::{FfcPair, with_context};

/// Creates an opaque two-tensor pair handle.
#[pd_host_function(name = "flint::pair::new")]
pub(super) fn pair_new_impl(local: i64, global: i64) -> VmResult<i64> {
    with_context(|context| {
        context.tensor(local)?;
        if global != 0 {
            context.tensor(global)?;
        }
        Ok(context.insert_pair(FfcPair { local, global }))
    })
}

/// Returns the local tensor handle from a pair.
#[pd_host_function(name = "flint::pair::local")]
pub(super) fn pair_local_impl(pair: i64) -> VmResult<i64> {
    with_context(|context| Ok(context.pair(pair)?.local))
}

/// Returns the global tensor handle from a pair.
#[pd_host_function(name = "flint::pair::global")]
pub(super) fn pair_global_impl(pair: i64) -> VmResult<i64> {
    with_context(|context| Ok(context.pair(pair)?.global))
}

# koharu-diffusion (Flint bindgen width patch)

Vendored from locked koharu commit `3a832b5dda50ce4d86d8a703535b46d6592e24e0`
(`git+https://github.com/mayocream/koharu.git?branch=refactor/0705`).

bindgen/clang on this host emit `sample_method_t` (and sibling C enums) as
`c_uint` / `u32`. Upstream wrappers hard-code `#[repr(i32)]` and `TryFrom<i32>`,
so the locked crate does not compile against the generated sys bindings.

This tree is that crate plus a minimal FFI-width fix in `src/enums.rs`.
`koharu-diffusion-sys` stays on the locked git revision. Do not edit Cargo's
git checkout.

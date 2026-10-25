# `llama-crab-sys`

Low-level FFI bindings to [`llama.cpp`](https://github.com/ggml-org/llama.cpp), `ggml`, `gguf` and `mtmd`.

Generated at build time with [`bindgen`](https://docs.rs/bindgen) over the
bundled C headers and built with CMake. This crate is **unsafe by design**:
every public item is a thin `extern "C"` wrapper.

Most users should depend on [`llama-crab`](https://crates.io/crates/llama-crab)
instead. Reach for this crate only when you need direct access to raw
llama.cpp symbols.

## Features

| Feature | Description |
| --- | --- |
| `openmp` | OpenMP CPU backend. Enabled by default. |
| `metal` | Apple Metal. Enabled by default on macOS aarch64. |
| `cuda` / `cuda-no-vmm` | NVIDIA CUDA backends. |
| `vulkan` | Vulkan backend. |
| `rocm` | AMD ROCm/HIP backend. |
| `opencl` | OpenCL backend. |
| `kleidiai` | KleidiAI CPU kernels for Arm mobile targets. |
| `mtmd` | Multimodal (vision + audio) helpers. |
| `common` | Builds `libcommon.a` for chat templates and JSON schema helpers. |
| `llguidance` | `llguidance` sampler integration. |
| `dynamic-link` | Link against `libllama` as a shared object. |
| `system-ggml` | Use a system GGML installation. |
| `dynamic-backends` | Load GGML backends at runtime. |

## Resources

- [API reference (docs.rs)](https://docs.rs/llama-crab-sys)
- [Workspace README](../../README.md)

## License

Licensed under the [MIT License](../../LICENSE-MIT).

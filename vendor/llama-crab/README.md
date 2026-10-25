# `llama-crab`

Safe, ergonomic and complete Rust bindings for [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

This is the main crate of the `llama-crab` workspace. Most applications
should depend on this crate; use [`llama-crab-sys`](../llama-crab-sys) only
when you need direct access to raw llama.cpp symbols.

## Quickstart

```toml
[dependencies]
llama-crab = "0.1"
```

```rust,no_run
use llama_crab::{Llama, LlamaParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut llama = Llama::load(
        LlamaParams::new("models/model.gguf")
            .with_n_ctx(2048)
            .with_n_gpu_layers(99),
    )?;

    let response = llama.create_completion("The capital of France is", 32)?;
    println!("{}", response.text);

    Ok(())
}
```

### Loading a model from Hugging Face

Pass a Hugging Face repository id (e.g. `TheBloke/Llama-2-7B-Chat-GGUF`) directly to
`LlamaParams::new`; the library will download the GGUF to the official HF cache and load it.
For repos with multiple `.gguf` files, specify the filename via `with_hf_filename`:

```rust,no_run
use llama_crab::{Llama, LlamaParams};

let mut llama = Llama::load(
    LlamaParams::new("TheBloke/Llama-2-7B-Chat-GGUF")
        .with_hf_filename("llama-2-7b-chat.Q4_K_M.gguf")
        .with_n_ctx(2048),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The model is cached at `~/.cache/huggingface/hub` (or `$HF_HOME/hub` if set). Set
`HF_TOKEN` for gated repos. Requires the `hf-hub` cargo feature:

```toml
[dependencies]
llama-crab = { version = "0.1", features = ["hf-hub"] }
```

## Features

- Text completion, chat completion, infill, embeddings and reranking.
- Sampling chains, grammar-constrained decoding, JSON-Schema to GBNF.
- Tool-call parsing for ChatML, Mistral, Llama 3, Functionary and plain JSON.
- Multimodal support (vision and audio) through `mtmd`.
- Hardware backends for CPU, Metal, CUDA, Vulkan, ROCm, OpenCL and KleidiAI.
- Mobile presets and packaging profiles for Android and iOS.

For the full feature table, backend flags and mobile build details, see the
[user guide](https://llama-crab.nlp.rocks/).

## Resources

- [API reference (0.1.7)](https://docs.rs/llama-crab/0.1.7/llama_crab/) — `rustdoc` for the current release.
- [API reference (latest)](https://docs.rs/llama-crab/latest/llama_crab/) — `rustdoc` for the latest version.
- [User guide](https://llama-crab.nlp.rocks/)
- [Examples repository](https://github.com/DominguesM/llama-crab-examples)
- [Workspace README](../../README.md)

## License

Licensed under the [MIT License](../../LICENSE-MIT).

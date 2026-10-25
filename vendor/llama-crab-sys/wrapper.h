// Master header included by bindgen.
//
// Optional local helper headers are only included here after they are backed by
// upstream implementations. Placeholder llama_rs_* shims stay out of bindgen.

#pragma once

#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>

#if defined(LLAMA_CRAB_HAS_MTMD)
#  include <mtmd.h>
#  include <mtmd-helper.h>
#endif

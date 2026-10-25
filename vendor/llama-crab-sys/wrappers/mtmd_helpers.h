// Reserved multimodal (mtmd) helper bridge. Upstream mtmd_* and mtmd_helper_*
// symbols are exposed directly; this local llama_rs_* helper is not.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// v0.1.4 placeholder. Returns non-zero if compiled directly.
int llama_rs_mtmd_init_helpers(void);

#ifdef __cplusplus
}
#endif

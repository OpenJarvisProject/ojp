// Reserved C-ABI bridge for `json_schema_to_grammar`.
// Not included by wrapper.h until it is backed by upstream implementation.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Convert a JSON Schema (string) into a GBNF grammar (string).
// Returns 0 on success and writes the grammar to `*out_grammar`.
// The caller must `free(*out_grammar)` when done.
//
// `force_gbnf` (when non-zero) keeps the grammar in the canonical
// GBNF form even when the input is simple enough to be expressed without
// backtracking. In v0.1.4 this returns non-zero unless the upstream
// implementation is explicitly compiled in.
int32_t llama_rs_json_schema_to_grammar(
    const char * json_schema,
    int32_t force_gbnf,
    char ** out_grammar
);

#ifdef __cplusplus
}
#endif

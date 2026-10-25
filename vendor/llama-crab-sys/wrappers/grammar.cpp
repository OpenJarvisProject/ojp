// Small C bridge kept out of bindgen in v0.1.4.
// It only succeeds when explicitly compiled with the upstream schema converter.

#include "grammar.h"

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(LLAMA_CRAB_HAS_COMMON_SCHEMA_LIB)
#  include "nlohmann/json.hpp"
#  include "common/json-schema-to-grammar.h"

extern "C" int32_t llama_rs_json_schema_to_grammar(
    const char * json_schema,
    int32_t force_gbnf,
    char ** out_grammar
) {
    if (json_schema == nullptr || out_grammar == nullptr) {
        return 1;
    }
    try {
        auto schema = nlohmann::ordered_json::parse(json_schema);
        std::string gbnf = json_schema_to_grammar(schema, static_cast<bool>(force_gbnf));
        char * buf = static_cast<char *>(std::malloc(gbnf.size() + 1));
        if (!buf) {
            return 2;
        }
        std::memcpy(buf, gbnf.data(), gbnf.size());
        buf[gbnf.size()] = '\0';
        *out_grammar = buf;
        return 0;
    } catch (const std::exception & e) {
        return 3;
    } catch (...) {
        return 4;
    }
}
#else
extern "C" int32_t llama_rs_json_schema_to_grammar(
    const char * json_schema,
    int32_t force_gbnf,
    char ** out_grammar
) {
    (void)json_schema;
    (void)force_gbnf;
    if (out_grammar != nullptr) {
        *out_grammar = nullptr;
    }
    return 4;
}
#endif

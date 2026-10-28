#include "engine.h"

const char *engine_version(void) { return "0.0.1"; }

void engine_add_f32(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

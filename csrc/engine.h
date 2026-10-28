#ifndef ENGINE_H
#define ENGINE_H

#include <stddef.h>

/* Fase 0: smoke test della toolchain Python <-> C. */
const char *engine_version(void);
void engine_add_f32(const float *a, const float *b, float *out, size_t n);

#endif

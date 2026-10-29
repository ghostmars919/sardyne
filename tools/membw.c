/* Banda di lettura della RAM con 1..N thread (stile STREAM, solo lettura).
 *
 * Il decode legge tutti i pesi a ogni token: la banda di lettura è il suo limite teorico
 * (roofline). Uso: build/membw [MB] [thread massimi]
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../csrc/threadpool.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
__attribute__((target("avx2"))) static float sum_avx2(const float *p, size_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_load_ps(p + i));
        a1 = _mm256_add_ps(a1, _mm256_load_ps(p + i + 8));
        a2 = _mm256_add_ps(a2, _mm256_load_ps(p + i + 16));
        a3 = _mm256_add_ps(a3, _mm256_load_ps(p + i + 24));
    }
    float out[8];
    _mm256_storeu_ps(out, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
}
#endif

static float sum_scalar(const float *p, size_t n) {
    float s[8] = {0};
    for (size_t i = 0; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; k++) s[k] += p[i + k];
    return s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
}

typedef struct {
    const float *buf;
    size_t n;
    float result[256];
    int simd;
} job;

static void worker(void *arg, int ith, int nth) {
    job *j = arg;
    size_t chunk = j->n / nth / 32 * 32;
    const float *p = j->buf + chunk * ith;
#if defined(__x86_64__) || defined(__i386__)
    j->result[ith] = j->simd ? sum_avx2(p, chunk) : sum_scalar(p, chunk);
#else
    j->result[ith] = sum_scalar(p, chunk);
#endif
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 512;
    int max_threads = argc > 2 ? atoi(argv[2]) : tp_default_threads();
    if (mb < 16 || max_threads < 1 || max_threads > 256) {
        fprintf(stderr, "uso: %s [MB >= 16] [thread 1..256]\n", argv[0]);
        return 2;
    }
    size_t n = mb * (1u << 20) / sizeof(float);
    float *buf;
    if (posix_memalign((void **)&buf, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "memoria esaurita\n");
        return 1;
    }
    for (size_t i = 0; i < n; i++) buf[i] = 1.0f; /* tocca tutte le pagine prima di misurare */

    int simd = 0;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    simd = __builtin_cpu_supports("avx2");
#endif
    printf("buffer %zu MB (molto più grande della cache L3), lettura %s\n", mb, simd ? "AVX2" : "scalare");
    printf("  thread   GB/s (migliore di 5)\n");
    for (int t = 1; t <= max_threads; t++) {
        threadpool *p = tp_new(t);
        job j = {buf, n, {0}, simd};
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now();
            tp_run(p, worker, &j);
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        double bytes = (double)(n / t / 32 * 32) * t * sizeof(float);
        printf("  %6d   %6.2f\n", t, bytes / best / 1e9);
        tp_free(p);
    }
    free(buf);
    return 0;
}

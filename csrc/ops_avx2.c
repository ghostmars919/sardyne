/* Kernel AVX2 + FMA + F16C (x86 da Haswell in poi).
 *
 * Ogni funzione è compilata per quelle estensioni con __attribute__((target)), anche se il resto
 * della libreria non lo è: ops.c le usa solo dopo aver verificato la CPU, quindi la stessa
 * libreria gira anche su CPU più vecchie (con i kernel scalari).
 *
 * Un registro ymm contiene 8 float. Una FMA (a*b + c) su 8 float per ciclo, con 2 unità FMA per
 * core, sono 32 operazioni fp32 per ciclo: 30 volte il codice scalare con una catena di dipendenze.
 */
#include "ops_internal.h"

#ifdef OPS_HAVE_AVX2

#include <immintrin.h>
#include <string.h>

#include "ops.h"

#define TARGET __attribute__((target("avx2,fma,f16c")))

int op_cpu_has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

/* Somma degli 8 float di un registro. */
TARGET static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* Caricamento di 8 pesi convertiti in fp32, per ciascun tipo. */
TARGET static inline __m256 load_f32(const float *p) { return _mm256_loadu_ps(p); }
TARGET static inline __m256 load_f16(const uint16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); /* F16C: conversione in hardware */
}
TARGET static inline __m256 load_bf16(const uint16_t *p) {
    __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)); /* bf16 = 16 bit alti del float32 */
}

static inline float elem_f32(const float *p, int i) { return p[i]; }
static inline float elem_f16(const uint16_t *p, int i) { return op_f16_to_f32(p[i]); }
static inline float elem_bf16(const uint16_t *p, int i) {
    uint32_t bits = (uint32_t)p[i] << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Per ogni tipo di peso: un prodotto scalare (decode) e un micro-kernel 2 righe × 4 token (prefill).
 *
 * dot: 4 accumulatori da 8 float, 32 elementi per iterazione. La latenza di una FMA è di 5 cicli:
 * con un solo accumulatore ogni FMA aspetterebbe la precedente.
 *
 * block_2x4: ogni peso caricato (e convertito) serve 4 token, ogni attivazione caricata serve
 * 2 righe; 8 accumulatori indipendenti. Per 8 FMA bastano 6 caricamenti invece di 16. */
#define DEFINE_AVX2_KERNELS(S, T)                                                                      \
    TARGET static float dot_##S(const float *x, const T *w, int n) {                                  \
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;                                    \
        int i = 0;                                                                                     \
        for (; i + 32 <= n; i += 32) {                                                                 \
            a0 = _mm256_fmadd_ps(load_##S(w + i), _mm256_loadu_ps(x + i), a0);                         \
            a1 = _mm256_fmadd_ps(load_##S(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);                 \
            a2 = _mm256_fmadd_ps(load_##S(w + i + 16), _mm256_loadu_ps(x + i + 16), a2);               \
            a3 = _mm256_fmadd_ps(load_##S(w + i + 24), _mm256_loadu_ps(x + i + 24), a3);               \
        }                                                                                              \
        for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(load_##S(w + i), _mm256_loadu_ps(x + i), a0);  \
        float s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));                   \
        for (; i < n; i++) s += x[i] * elem_##S(w, i);                                                 \
        return s;                                                                                      \
    }                                                                                                  \
                                                                                                       \
    TARGET static void block_2x4_##S(float *out, const float *x, const T *w0, const T *w1, int n_in,   \
                                     int n_out) {                                                      \
        const float *x0 = x, *x1 = x + n_in, *x2 = x + 2 * (size_t)n_in, *x3 = x + 3 * (size_t)n_in;   \
        __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;                             \
        __m256 c10 = c00, c11 = c00, c12 = c00, c13 = c00;                                             \
        int i = 0;                                                                                     \
        for (; i + 8 <= n_in; i += 8) {                                                                \
            __m256 b0 = load_##S(w0 + i), b1 = load_##S(w1 + i), a;                                    \
            a = _mm256_loadu_ps(x0 + i);                                                               \
            c00 = _mm256_fmadd_ps(b0, a, c00);                                                         \
            c10 = _mm256_fmadd_ps(b1, a, c10);                                                         \
            a = _mm256_loadu_ps(x1 + i);                                                               \
            c01 = _mm256_fmadd_ps(b0, a, c01);                                                         \
            c11 = _mm256_fmadd_ps(b1, a, c11);                                                         \
            a = _mm256_loadu_ps(x2 + i);                                                               \
            c02 = _mm256_fmadd_ps(b0, a, c02);                                                         \
            c12 = _mm256_fmadd_ps(b1, a, c12);                                                         \
            a = _mm256_loadu_ps(x3 + i);                                                               \
            c03 = _mm256_fmadd_ps(b0, a, c03);                                                         \
            c13 = _mm256_fmadd_ps(b1, a, c13);                                                         \
        }                                                                                              \
        float s[2][4] = {{hsum(c00), hsum(c01), hsum(c02), hsum(c03)},                                 \
                         {hsum(c10), hsum(c11), hsum(c12), hsum(c13)}};                                \
        for (; i < n_in; i++) {                                                                        \
            float e0 = elem_##S(w0, i), e1 = elem_##S(w1, i);                                          \
            const float *xs[4] = {x0, x1, x2, x3};                                                     \
            for (int t = 0; t < 4; t++) {                                                              \
                s[0][t] += e0 * xs[t][i];                                                              \
                s[1][t] += e1 * xs[t][i];                                                              \
            }                                                                                          \
        }                                                                                              \
        for (int t = 0; t < 4; t++) {                                                                  \
            out[(size_t)t * n_out] = s[0][t];                                                          \
            out[(size_t)t * n_out + 1] = s[1][t];                                                      \
        }                                                                                              \
    }                                                                                                  \
                                                                                                       \
    TARGET void op_matmul_rows_avx2_##S(float *out, const float *x, const void *wv, int n_tok, int n_in, \
                                        int n_out, int row0, int row1) {                               \
        const T *w = wv;                                                                               \
        int j = row0;                                                                                  \
        /* Righe a coppie (all'esterno: ogni coppia di righe passa dalla RAM una volta sola) e token   \
         * a blocchi di 4 (all'interno: le attivazioni restano nella cache). */                         \
        if (n_tok >= 4) {                                                                              \
            for (; j + 2 <= row1; j += 2) {                                                            \
                const T *w0 = w + (size_t)j * n_in, *w1 = w0 + n_in;                                   \
                int t = 0;                                                                             \
                for (; t + 4 <= n_tok; t += 4)                                                         \
                    block_2x4_##S(out + (size_t)t * n_out + j, x + (size_t)t * n_in, w0, w1, n_in, n_out); \
                for (; t < n_tok; t++) {                                                               \
                    out[(size_t)t * n_out + j] = dot_##S(x + (size_t)t * n_in, w0, n_in);              \
                    out[(size_t)t * n_out + j + 1] = dot_##S(x + (size_t)t * n_in, w1, n_in);          \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
        for (; j < row1; j++)                                                                          \
            for (int t = 0; t < n_tok; t++)                                                            \
                out[(size_t)t * n_out + j] = dot_##S(x + (size_t)t * n_in, w + (size_t)j * n_in, n_in); \
    }

DEFINE_AVX2_KERNELS(f32, float)
DEFINE_AVX2_KERNELS(f16, uint16_t)
DEFINE_AVX2_KERNELS(bf16, uint16_t)

TARGET float op_dot_f32_avx2(const float *a, const float *b, int n) { return dot_f32(a, b, n); }

TARGET void op_axpy_avx2(float *y, float a, const float *x, int n) {
    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}

#endif

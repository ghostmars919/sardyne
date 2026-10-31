#define _POSIX_C_SOURCE 200809L

#include "ops.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "ops_internal.h"
#include "quant.h"

/* ---------------------------------------------------------------- conversioni */

float op_f16_to_f32(uint16_t h) {
    /* IEEE 754 half: 1 bit di segno, 5 di esponente (bias 15), 10 di mantissa. */
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* ±0 */
        } else {
            /* Subnormale (mant · 2^-24): lo si normalizza spostando la mantissa. */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13); /* inf o NaN */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t op_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, a = x & 0x7FFFFFFF;
    if (a > 0x7F800000) return (uint16_t)(sign | 0x7E00);   /* NaN */
    if (a >= 0x477FF000) return (uint16_t)(sign | 0x7C00);  /* >= 65520: arrotonda a inf */
    if (a < 0x38800000) {
        /* Sotto 2^-14 il risultato è subnormale: valore / 2^-24 arrotondato al pari. Moltiplicare
         * per una potenza di 2 è esatto; 1,5·2^23 sposta l'intero nei bit bassi della mantissa. */
        float v;
        memcpy(&v, &a, 4);
        float t = v * 16777216.0f + 12582912.0f;
        uint32_t m;
        memcpy(&m, &t, 4);
        return (uint16_t)(sign | ((m & 0x007FFFFF) - 0x00400000));
    }
    /* Normale: esponente con bias 15 invece di 127 e mantissa da 23 a 10 bit, al pari. */
    a += 0xC8000FFF + ((a >> 13) & 1);
    return (uint16_t)(sign | (a >> 13));
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16; /* bf16 = 16 bit alti di un float32 */
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Tabella di tutti i 65536 valori f16: una lettura invece di una conversione per peso
 * (serve al percorso scalare; con F16C la conversione la fa la CPU). */
static float F16_TABLE[1 << 16];

/* ---------------------------------------------------------------- kernel scalari */

/* Quattro accumulatori indipendenti: ogni somma non deve più aspettare la precedente, e la
 * CPU può tenere in volo più moltiplicazioni-somme contemporaneamente. */
#define DEFINE_SCALAR_DOT(NAME, T, ELEM)                                                     \
    static float NAME(const float *x, const T *w, int n) {                                  \
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;                                    \
        int i = 0;                                                                          \
        for (; i + 4 <= n; i += 4) {                                                        \
            s0 += x[i] * ELEM(w[i]);                                                        \
            s1 += x[i + 1] * ELEM(w[i + 1]);                                                \
            s2 += x[i + 2] * ELEM(w[i + 2]);                                                \
            s3 += x[i + 3] * ELEM(w[i + 3]);                                                \
        }                                                                                   \
        for (; i < n; i++) s0 += x[i] * ELEM(w[i]);                                         \
        return (s0 + s1) + (s2 + s3);                                                       \
    }

#define ELEM_F32(v) (v)
#define ELEM_F16(v) F16_TABLE[v]
#define ELEM_BF16(v) bf16_to_f32(v)
DEFINE_SCALAR_DOT(dot_scalar_f32, float, ELEM_F32)
DEFINE_SCALAR_DOT(dot_scalar_f16, uint16_t, ELEM_F16)
DEFINE_SCALAR_DOT(dot_scalar_bf16, uint16_t, ELEM_BF16)

#define DEFINE_SCALAR_MATMUL(NAME, T, DOT)                                                          \
    static void NAME(float *out, const float *x, const void *wv, int n_tok, int n_in, int n_out,     \
                     int row0, int row1) {                                                          \
        const T *w = wv;                                                                            \
        /* Righe all'esterno, token all'interno: ogni riga dei pesi viene letta dalla RAM una       \
         * volta sola e riusata dalla cache per tutti i token del blocco. */                        \
        for (int j = row0; j < row1; j++)                                                           \
            for (int t = 0; t < n_tok; t++)                                                         \
                out[(size_t)t * n_out + j] = DOT(x + (size_t)t * n_in, w + (size_t)j * n_in, n_in); \
    }

DEFINE_SCALAR_MATMUL(matmul_scalar_f32, float, dot_scalar_f32)
DEFINE_SCALAR_MATMUL(matmul_scalar_f16, uint16_t, dot_scalar_f16)
DEFINE_SCALAR_MATMUL(matmul_scalar_bf16, uint16_t, dot_scalar_bf16)

static float dot_f32_scalar(const float *a, const float *b, int n) { return dot_scalar_f32(a, b, n); }

static void axpy_scalar(float *y, float a, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

/* ---------------------------------------------------------------- scelta dei kernel */

/* Indice dei tipi quantizzati nella tabella dei prodotti scalari. */
static int qindex(int type) {
    switch (type) {
    case OP_Q8_0: return 0;
    case OP_Q4_0: return 1;
    case OP_Q5_0: return 2;
    case OP_Q4_K: return 3;
    case OP_Q6_K: return 4;
    default: return -1;
    }
}

static struct {
    op_matmul_rows_fn matmul[3]; /* F32, F16, BF16 */
    quant_dot_fn qdot[5];        /* Q8_0, Q4_0, Q5_0, Q4_K, Q6_K (ordine di qindex) */
    quant_matmul_rows_fn qmatmul[5]; /* NULL: matmul generica, un prodotto scalare per (riga, token) */
    float (*dot)(const float *, const float *, int);
    void (*axpy)(float *, float, const float *, int);
    int simd;
} K;

static int simd_available(void) {
#ifdef OPS_HAVE_AVX2
    return op_cpu_has_avx2();
#else
    return 0;
#endif
}

static void select_kernels(int want_simd) {
    K.matmul[0] = matmul_scalar_f32;
    K.matmul[1] = matmul_scalar_f16;
    K.matmul[2] = matmul_scalar_bf16;
    K.dot = dot_f32_scalar;
    K.axpy = axpy_scalar;
    K.qdot[0] = quant_dot_q8_0;
    K.qdot[1] = quant_dot_q4_0;
    K.qdot[2] = quant_dot_q5_0;
    K.qdot[3] = quant_dot_q4_K;
    K.qdot[4] = quant_dot_q6_K;
    for (int i = 0; i < 5; i++) K.qmatmul[i] = NULL;
    K.simd = 0;
#ifdef OPS_HAVE_AVX2
    if (want_simd && simd_available()) {
        K.matmul[0] = op_matmul_rows_avx2_f32;
        K.matmul[1] = op_matmul_rows_avx2_f16;
        K.matmul[2] = op_matmul_rows_avx2_bf16;
        K.dot = op_dot_f32_avx2;
        K.axpy = op_axpy_avx2;
        K.qdot[0] = quant_dot_q8_0_avx2;
        K.qdot[1] = quant_dot_q4_0_avx2;
        K.qdot[2] = quant_dot_q5_0_avx2;
        K.qdot[3] = quant_dot_q4_K_avx2;
        K.qdot[4] = quant_dot_q6_K_avx2;
        K.qmatmul[0] = quant_matmul_rows_q8_0_avx2;
        K.qmatmul[1] = quant_matmul_rows_q4_0_avx2;
        K.qmatmul[2] = quant_matmul_rows_q5_0_avx2;
        K.qmatmul[3] = quant_matmul_rows_q4_K_avx2;
        K.qmatmul[4] = quant_matmul_rows_q6_K_avx2;
        K.simd = 1;
    }
#else
    (void)want_simd;
#endif
}

static pthread_once_t init_once = PTHREAD_ONCE_INIT;

static void init(void) {
    for (uint32_t i = 0; i < (1u << 16); i++) F16_TABLE[i] = op_f16_to_f32((uint16_t)i);
    const char *env = getenv("ENGINE_NO_SIMD");
    select_kernels(!(env && env[0] && strcmp(env, "0") != 0));
}

void op_init(void) { pthread_once(&init_once, init); }

int op_simd(void) {
    op_init();
    return K.simd;
}

int op_set_simd(int enable) {
    /* Da usare solo quando nessun kernel è in esecuzione (serve ai test e ai benchmark). */
    op_init();
    select_kernels(enable);
    return K.simd;
}

size_t op_type_size(int type) {
    switch (type) {
    case OP_F32: return 4;
    case OP_F16: case OP_BF16: return 2;
    default: return 0;
    }
}

int op_block_size(int type) {
    switch (type) {
    case OP_F32: case OP_F16: case OP_BF16: return 1;
    case OP_Q8_0: case OP_Q4_0: case OP_Q5_0: return QK_0;
    case OP_Q4_K: case OP_Q6_K: case OP_Q8_K: return QK_K;
    default: return 0;
    }
}

static size_t block_bytes(int type) {
    switch (type) {
    case OP_F32: return 4;
    case OP_F16: case OP_BF16: return 2;
    case OP_Q8_0: return sizeof(block_q8_0);
    case OP_Q4_0: return sizeof(block_q4_0);
    case OP_Q5_0: return sizeof(block_q5_0);
    case OP_Q4_K: return sizeof(block_q4_K);
    case OP_Q6_K: return sizeof(block_q6_K);
    case OP_Q8_K: return sizeof(block_q8_K);
    default: return 0;
    }
}

int op_type_supported(int type) { return type != OP_Q8_K && block_bytes(type) != 0; }

size_t op_row_size(int type, int n) {
    int b = op_block_size(type);
    return b ? (size_t)(n / b) * block_bytes(type) : 0;
}

int op_vec_dot_type(int wtype) {
    switch (wtype) {
    case OP_Q8_0: case OP_Q4_0: case OP_Q5_0: return OP_Q8_0;
    case OP_Q4_K: case OP_Q6_K: return OP_Q8_K;
    default: return OP_F32;
    }
}

void op_f16_to_f32_n(const uint16_t *x, float *out, size_t n) {
    op_init();
    for (size_t i = 0; i < n; i++) out[i] = F16_TABLE[x[i]];
}

void op_row_to_f32(const void *row, int type, float *out, size_t n) {
    if (type == OP_F32) {
        memcpy(out, row, n * sizeof(float));
    } else if (type == OP_F16) {
        op_f16_to_f32_n(row, out, n);
    } else if (qindex(type) >= 0) {
        quant_dequant_row(type, row, out, (int)n);
    } else {
        const uint16_t *r = row;
        for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(r[i]);
    }
}

/* ---------------------------------------------------------------- matmul e prodotti scalari */

void op_matmul_rows(float *out, const float *x, const void *w, int wtype, int n_tok, int n_in, int n_out,
                    int row0, int row1) {
    op_init();
    int k = wtype == OP_F32 ? 0 : wtype == OP_F16 ? 1 : 2;
    K.matmul[k](out, x, w, n_tok, n_in, n_out, row0, row1);
}

void op_quantize_rows(int vtype, const float *x, void *xq, int n_in, int t0, int t1) {
    size_t row = op_row_size(vtype, n_in);
    for (int t = t0; t < t1; t++) {
        void *dst = (uint8_t *)xq + (size_t)t * row;
        if (vtype == OP_Q8_0) quant_q8_0(x + (size_t)t * n_in, dst, n_in);
        else quant_q8_K(x + (size_t)t * n_in, dst, n_in);
    }
}

void op_matmul_q_rows(float *out, const void *xq, const void *w, int wtype, int n_tok, int n_in, int n_out,
                      int row0, int row1) {
    op_init();
    int k = qindex(wtype);
    if (K.qmatmul[k]) { /* micro-kernel che riusa ogni blocco di pesi per 4 token */
        K.qmatmul[k](out, xq, w, n_tok, n_in, n_out, row0, row1);
        return;
    }
    quant_dot_fn dot = K.qdot[k];
    size_t wrow = op_row_size(wtype, n_in), xrow = op_row_size(op_vec_dot_type(wtype), n_in);
    /* Come per i float: righe all'esterno (lette dalla RAM una volta), token all'interno. */
    for (int j = row0; j < row1; j++) {
        const uint8_t *wj = (const uint8_t *)w + (size_t)j * wrow;
        for (int t = 0; t < n_tok; t++)
            out[(size_t)t * n_out + j] = dot(wj, (const uint8_t *)xq + (size_t)t * xrow, n_in);
    }
}

float op_vec_dot_q(int wtype, const void *w, const void *xq, int n) {
    op_init();
    return K.qdot[qindex(wtype)](w, xq, n);
}

void op_matmul(float *out, const float *x, const void *w, int wtype, int n_tok, int n_in, int n_out) {
    int vt = op_vec_dot_type(wtype);
    if (vt == OP_F32) {
        op_matmul_rows(out, x, w, wtype, n_tok, n_in, n_out, 0, n_out);
        return;
    }
    void *xq = malloc(op_row_size(vt, n_in) * (size_t)n_tok);
    if (!xq) return;
    op_quantize_rows(vt, x, xq, n_in, 0, n_tok);
    op_matmul_q_rows(out, xq, w, wtype, n_tok, n_in, n_out, 0, n_out);
    free(xq);
}

void op_add_bias(float *x, const float *b, int n_tok, int n) {
    for (int t = 0; t < n_tok; t++)
        for (int i = 0; i < n; i++) x[(size_t)t * n + i] += b[i];
}

float op_dot_f32(const float *a, const float *b, int n) {
    op_init();
    return K.dot(a, b, n);
}

void op_axpy(float *y, float a, const float *x, int n) {
    op_init();
    K.axpy(y, a, x, n);
}

/* ---------------------------------------------------------------- normalizzazione e attivazioni */

void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0.0; /* somma in double: n può arrivare a migliaia di termini */
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

void op_softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max); /* sottrarre il massimo evita l'overflow */
        sum += x[i];
    }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void op_silu_mul(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

/* ---------------------------------------------------------------- RoPE */

void op_rope_table(float *cos_out, float *sin_out, int n_pos, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        /* In double come in sardyne/ref/ops.py::rope_cos_sin. */
        double inv_freq = 1.0 / pow((double)theta, (double)(2 * i) / head_dim);
        for (int p = 0; p < n_pos; p++) {
            double angle = (double)p * inv_freq;
            cos_out[(size_t)p * half + i] = (float)cos(angle);
            sin_out[(size_t)p * half + i] = (float)sin(angle);
        }
    }
}

void op_rope(float *x, int n_heads, int head_dim, const float *cos, const float *sin, int interleaved) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *v = x + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            /* Coppia (a, b) ruotata dell'angolo i-esimo: (a cos - b sin, a sin + b cos). */
            int ia = interleaved ? 2 * i : i;
            int ib = interleaved ? 2 * i + 1 : i + half;
            float a = v[ia], b = v[ib];
            v[ia] = a * cos[i] - b * sin[i];
            v[ib] = a * sin[i] + b * cos[i];
        }
    }
}

/* ---------------------------------------------------------------- attenzione */

void op_attention_head(float *out, const float *q, const float *k, const float *v, int head_dim, int n_kv,
                       float *scores) {
    op_init();
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int j = 0; j < n_kv; j++) scores[j] = K.dot(q, k + (size_t)j * head_dim, head_dim) * scale;
    op_softmax(scores, n_kv);
    memset(out, 0, (size_t)head_dim * sizeof(float));
    for (int j = 0; j < n_kv; j++) K.axpy(out, scores[j], v + (size_t)j * head_dim, head_dim);
}

void op_attention(float *out, const float *q, const float *k_cache, const float *v_cache,
                  int n_head, int n_kv_head, int head_dim, int n_kv, int stride, float *scores) {
    int n_rep = n_head / n_kv_head;
    for (int h = 0; h < n_head; h++) {
        size_t kv = (size_t)(h / n_rep) * stride * head_dim;
        op_attention_head(out + (size_t)h * head_dim, q + (size_t)h * head_dim, k_cache + kv, v_cache + kv,
                          head_dim, n_kv, scores);
    }
}

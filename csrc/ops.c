#define _POSIX_C_SOURCE 200809L

#include "ops.h"

#include <math.h>
#include <pthread.h>
#include <string.h>

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

/* Tabella di tutti i 65536 valori f16: una lettura invece di una conversione per peso. */
static float F16_TABLE[1 << 16];
static pthread_once_t f16_once = PTHREAD_ONCE_INIT;

static void build_f16_table(void) {
    for (uint32_t i = 0; i < (1u << 16); i++) F16_TABLE[i] = op_f16_to_f32((uint16_t)i);
}

void op_init(void) { pthread_once(&f16_once, build_f16_table); }

static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16; /* bf16 = 16 bit alti di un float32 */
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void op_f16_to_f32_n(const uint16_t *x, float *out, size_t n) {
    op_init();
    for (size_t i = 0; i < n; i++) out[i] = F16_TABLE[x[i]];
}

size_t op_type_size(int type) {
    switch (type) {
    case OP_F32: return 4;
    case OP_F16: case OP_BF16: return 2;
    default: return 0;
    }
}

void op_row_to_f32(const void *row, int type, float *out, size_t n) {
    if (type == OP_F32) {
        memcpy(out, row, n * sizeof(float));
    } else if (type == OP_F16) {
        op_f16_to_f32_n(row, out, n);
    } else {
        const uint16_t *r = row;
        for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(r[i]);
    }
}

/* ---------------------------------------------------------------- prodotti scalari */

static float dot_f32(const float *x, const float *w, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * w[i];
    return s;
}

static float dot_f16(const float *x, const uint16_t *w, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * F16_TABLE[w[i]];
    return s;
}

static float dot_bf16(const float *x, const uint16_t *w, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * bf16_to_f32(w[i]);
    return s;
}

void op_matmul(float *out, const float *x, const void *w, int wtype, int n_tok, int n_in, int n_out) {
    op_init();
    size_t row_bytes = (size_t)n_in * op_type_size(wtype);
    /* Ciclo esterno sulle righe dei pesi, interno sui token: ogni riga viene letta dalla memoria
     * una sola volta e riusata (dalla cache della CPU) per tutti i token del blocco. È il motivo
     * per cui il prefill di T token costa molto meno di T passi di decode. */
    for (int j = 0; j < n_out; j++) {
        const void *row = (const uint8_t *)w + (size_t)j * row_bytes;
        for (int t = 0; t < n_tok; t++) {
            const float *xt = x + (size_t)t * n_in;
            float s;
            if (wtype == OP_F32) s = dot_f32(xt, row, n_in);
            else if (wtype == OP_F16) s = dot_f16(xt, row, n_in);
            else s = dot_bf16(xt, row, n_in);
            out[(size_t)t * n_out + j] = s;
        }
    }
}

void op_add_bias(float *x, const float *b, int n_tok, int n) {
    for (int t = 0; t < n_tok; t++)
        for (int i = 0; i < n; i++) x[(size_t)t * n + i] += b[i];
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
        /* In double come in engine/ref/ops.py::rope_cos_sin. */
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

void op_attention(float *out, const float *q, const float *k_cache, const float *v_cache,
                  int n_head, int n_kv_head, int head_dim, int n_kv, int stride, float *scores) {
    int n_rep = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < n_head; h++) {
        const float *qh = q + (size_t)h * head_dim;
        const float *kh = k_cache + (size_t)(h / n_rep) * stride * head_dim;
        const float *vh = v_cache + (size_t)(h / n_rep) * stride * head_dim;
        for (int j = 0; j < n_kv; j++) scores[j] = dot_f32(qh, kh + (size_t)j * head_dim, head_dim) * scale;
        op_softmax(scores, n_kv);
        float *oh = out + (size_t)h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int j = 0; j < n_kv; j++) {
            const float *vj = vh + (size_t)j * head_dim;
            float s = scores[j];
            for (int d = 0; d < head_dim; d++) oh[d] += s * vj[d];
        }
    }
}

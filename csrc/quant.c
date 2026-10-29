/* Kernel scalari per i formati quantizzati: la versione portabile e la reference per AVX2. */
#include "quant.h"

#include <math.h>
#include <string.h>

#include "ops.h"

#define F16(x) op_f16_to_f32(x)

/* ---------------------------------------------------------------- dequantizzazione (embedding) */

static void dequant_q8_0(const block_q8_0 *b, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        float d = F16(b[i].d);
        for (int j = 0; j < 32; j++) y[i * 32 + j] = b[i].qs[j] * d;
    }
}

static void dequant_q4_0(const block_q4_0 *b, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        float d = F16(b[i].d);
        for (int j = 0; j < 16; j++) {
            y[i * 32 + j] = ((b[i].qs[j] & 0x0F) - 8) * d;
            y[i * 32 + j + 16] = ((b[i].qs[j] >> 4) - 8) * d;
        }
    }
}

static void dequant_q5_0(const block_q5_0 *b, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        float d = F16(b[i].d);
        uint32_t qh;
        memcpy(&qh, b[i].qh, 4);
        for (int j = 0; j < 16; j++) {
            int x0 = ((b[i].qs[j] & 0x0F) | (((qh >> j) & 1) << 4)) - 16;
            int x1 = ((b[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
            y[i * 32 + j] = x0 * d;
            y[i * 32 + j + 16] = x1 * d;
        }
    }
}

static void dequant_q4_K(const block_q4_K *b, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        float d = F16(b[i].d), dmin = F16(b[i].dmin);
        const uint8_t *q = b[i].qs;
        for (int j = 0; j < 4; j++) { /* 64 elementi: 32 bassi (scala 2j) e 32 alti (scala 2j+1) */
            uint8_t sc, m;
            q4k_scale_min(2 * j, b[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            q4k_scale_min(2 * j + 1, b[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
        }
    }
}

/* Valori a 6 bit (−32..31) di un super-blocco Q6_K, nell'ordine degli elementi. */
static void q6k_values(const block_q6_K *b, int8_t *a) {
    const uint8_t *ql = b->ql, *qh = b->qh;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            a[n + l] = (int8_t)(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            a[n + l + 32] = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            a[n + l + 64] = (int8_t)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            a[n + l + 96] = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        ql += 64;
        qh += 32;
    }
}

static void dequant_q6_K(const block_q6_K *b, float *y, int nb) {
    int8_t a[256];
    for (int i = 0; i < nb; i++) {
        float d = F16(b[i].d);
        q6k_values(&b[i], a);
        for (int j = 0; j < 256; j++) y[i * 256 + j] = d * b[i].scales[j / 16] * a[j];
    }
}

void quant_dequant_row(int type, const void *src, float *dst, int n) {
    switch (type) {
    case OP_Q8_0: dequant_q8_0(src, dst, n / 32); break;
    case OP_Q4_0: dequant_q4_0(src, dst, n / 32); break;
    case OP_Q5_0: dequant_q5_0(src, dst, n / 32); break;
    case OP_Q4_K: dequant_q4_K(src, dst, n / 256); break;
    case OP_Q6_K: dequant_q6_K(src, dst, n / 256); break;
    default: break;
    }
}

/* ---------------------------------------------------------------- quantizzazione delle attivazioni */

void quant_q8_0(const float *x, void *dst, int n) {
    block_q8_0 *y = dst;
    for (int i = 0; i < n / 32; i++, x += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = op_f32_to_f16(d);
        for (int j = 0; j < 32; j++) y[i].qs[j] = (int8_t)roundf(x[j] * id);
    }
}

/* Arrotondamento al pari più vicino, come nearest_int di ggml (|v| < 2^22). */
static inline int nearest_int(float v) {
    float f = v + 12582912.0f; /* 1,5 · 2^23: la somma lascia l'intero nei bit della mantissa */
    int32_t i;
    memcpy(&i, &f, 4);
    return (i & 0x007FFFFF) - 0x00400000;
}

void quant_q8_K(const float *x, void *dst, int n) {
    block_q8_K *y = dst;
    for (int i = 0; i < n / 256; i++, x += 256) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 256; j++) {
            if (fabsf(x[j]) > amax) {
                amax = fabsf(x[j]);
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            memset(&y[i], 0, sizeof(y[i]));
            continue;
        }
        float iscale = -127.0f / max;
        for (int j = 0; j < 256; j++) {
            int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int k = 0; k < 16; k++) s += y[i].qs[g * 16 + k];
            y[i].bsums[g] = (int16_t)s;
        }
        y[i].d = 1.0f / iscale;
    }
}

/* ---------------------------------------------------------------- prodotti scalari */

float quant_dot_q8_0(const void *w, const void *x, int n) {
    const block_q8_0 *a = w, *b = x;
    float acc = 0.0f;
    for (int i = 0; i < n / 32; i++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += a[i].qs[j] * b[i].qs[j];
        acc += (float)s * (F16(a[i].d) * F16(b[i].d));
    }
    return acc;
}

float quant_dot_q4_0(const void *w, const void *x, int n) {
    const block_q4_0 *a = w;
    const block_q8_0 *b = x;
    float acc = 0.0f;
    for (int i = 0; i < n / 32; i++) {
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            s += ((a[i].qs[j] & 0x0F) - 8) * b[i].qs[j];
            s += ((a[i].qs[j] >> 4) - 8) * b[i].qs[j + 16];
        }
        acc += (float)s * (F16(a[i].d) * F16(b[i].d));
    }
    return acc;
}

float quant_dot_q5_0(const void *w, const void *x, int n) {
    const block_q5_0 *a = w;
    const block_q8_0 *b = x;
    float acc = 0.0f;
    for (int i = 0; i < n / 32; i++) {
        uint32_t qh;
        memcpy(&qh, a[i].qh, 4);
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            s += (((a[i].qs[j] & 0x0F) | (((qh >> j) & 1) << 4)) - 16) * b[i].qs[j];
            s += (((a[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16) * b[i].qs[j + 16];
        }
        acc += (float)s * (F16(a[i].d) * F16(b[i].d));
    }
    return acc;
}

float quant_dot_q4_K(const void *w, const void *x, int n) {
    const block_q4_K *a = w;
    const block_q8_K *b = x;
    float acc = 0.0f;
    for (int i = 0; i < n / 256; i++) {
        const uint8_t *q = a[i].qs;
        const int8_t *y = b[i].qs;
        int32_t isum = 0, msum = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc0, m0, sc1, m1;
            q4k_scale_min(2 * j, a[i].scales, &sc0, &m0);
            q4k_scale_min(2 * j + 1, a[i].scales, &sc1, &m1);
            int32_t s0 = 0, s1 = 0;
            for (int l = 0; l < 32; l++) {
                s0 += (q[l] & 0xF) * y[l];
                s1 += (q[l] >> 4) * y[l + 32];
            }
            isum += sc0 * s0 + sc1 * s1;
            /* minimi: −dmin·m·Σy, con Σy dai bsums (due gruppi da 16 per sotto-blocco) */
            msum += m0 * (b[i].bsums[4 * j] + b[i].bsums[4 * j + 1]) + m1 * (b[i].bsums[4 * j + 2] + b[i].bsums[4 * j + 3]);
            q += 32;
            y += 64;
        }
        acc = q4k_combine(acc, b[i].d, F16(a[i].d), F16(a[i].dmin), isum, msum);
    }
    return acc;
}

float quant_dot_q6_K(const void *w, const void *x, int n) {
    const block_q6_K *a = w;
    const block_q8_K *b = x;
    int8_t v[256];
    float acc = 0.0f;
    for (int i = 0; i < n / 256; i++) {
        q6k_values(&a[i], v);
        int32_t isum = 0;
        for (int g = 0; g < 16; g++) {
            int32_t s = 0;
            for (int k = 0; k < 16; k++) s += v[g * 16 + k] * b[i].qs[g * 16 + k];
            isum += a[i].scales[g] * s;
        }
        acc = q6k_combine(acc, b[i].d, F16(a[i].d), isum);
    }
    return acc;
}

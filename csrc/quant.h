#ifndef QUANT_H
#define QUANT_H

/* Formati quantizzati di ggml (Fase 5). Reference NumPy: engine/quant.py.
 *
 * Il prodotto peso·attivazione non dequantizza i pesi: le attivazioni di ogni token vengono
 * quantizzate una volta in int8 (Q8_0 per i pesi a blocchi da 32, Q8_K per i K-quant da 256),
 * poi per ogni blocco si calcola un prodotto scalare intero esatto e si applicano le scale in fp32.
 */

#include <stddef.h>
#include <stdint.h>

enum { QK_0 = 32, QK_K = 256 };

typedef struct {
    uint16_t d;     /* scala f16 */
    int8_t qs[32];
} block_q8_0;

typedef struct {
    uint16_t d;
    uint8_t qs[16]; /* nibble basso = elemento j, alto = elemento j + 16; valore q − 8 */
} block_q4_0;

typedef struct {
    uint16_t d;
    uint8_t qh[4];  /* bit e = quinto bit dell'elemento e */
    uint8_t qs[16]; /* come Q4_0; valore (q | bit5 << 4) − 16 */
} block_q5_0;

typedef struct {
    uint16_t d, dmin;   /* scale f16 delle scale e dei minimi */
    uint8_t scales[12]; /* 8 scale e 8 minimi a 6 bit (q4k_scale_min) */
    uint8_t qs[128];    /* gruppi di 32 byte: 4 bit bassi = 32 elementi, alti = i 32 successivi */
} block_q4_K;

typedef struct {
    uint8_t ql[128];   /* 4 bit bassi */
    uint8_t qh[64];    /* 2 bit alti */
    int8_t scales[16]; /* una scala ogni 16 elementi */
    uint16_t d;
} block_q6_K;

typedef struct {
    float d;
    int8_t qs[256];
    int16_t bsums[16]; /* somme dei qs a gruppi di 16: servono ai minimi di Q4_K */
} block_q8_K;

_Static_assert(sizeof(block_q8_0) == 34, "block_q8_0");
_Static_assert(sizeof(block_q4_0) == 18, "block_q4_0");
_Static_assert(sizeof(block_q5_0) == 22, "block_q5_0");
_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K");
_Static_assert(sizeof(block_q6_K) == 210, "block_q6_K");
_Static_assert(sizeof(block_q8_K) == 292, "block_q8_K");

/* Prodotto scalare di una riga di pesi con una riga di attivazioni quantizzate (n elementi). */
typedef float (*quant_dot_fn)(const void *w, const void *x, int n);

float quant_dot_q8_0(const void *w, const void *x, int n);
float quant_dot_q4_0(const void *w, const void *x, int n);
float quant_dot_q5_0(const void *w, const void *x, int n);
float quant_dot_q4_K(const void *w, const void *x, int n);
float quant_dot_q6_K(const void *w, const void *x, int n);

void quant_dequant_row(int type, const void *src, float *dst, int n);
void quant_q8_0(const float *x, void *dst, int n); /* n multiplo di 32 */
void quant_q8_K(const float *x, void *dst, int n); /* n multiplo di 256 */

/* Scale e minimi a 6 bit di Q4_K, sotto-blocco j = 0..7 (get_scale_min_k4 di ggml). */
static inline void q4k_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Combinazione finale di un super-blocco K-quant, condivisa da scalare e AVX2: la parte intera è
 * esatta, e con la stessa espressione in fp32 i due percorsi danno risultati identici bit per bit. */
static inline float q4k_combine(float acc, float yd, float d, float dmin, int32_t isum, int32_t msum) {
    float dall = yd * d, dmn = yd * dmin;
    return acc + (dall * (float)isum - dmn * (float)msum);
}

static inline float q6k_combine(float acc, float yd, float d, int32_t isum) {
    return acc + (yd * d) * (float)isum;
}

#endif

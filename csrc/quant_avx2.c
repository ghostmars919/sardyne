/* Prodotti scalari AVX2 per i pesi quantizzati (stesso schema di ops_avx2.c: target attribute,
 * scelti a runtime da ops.c).
 *
 * Il cuore è _mm256_maddubs_epi16: 32 byte senza segno × 32 byte con segno, sommati a coppie in
 * 16 interi a 16 bit; poi _mm256_madd_epi16 con 1 (o con la scala del sotto-blocco) somma a coppie
 * in 8 interi a 32 bit. 32 pesi in 2 istruzioni, tutto in aritmetica intera esatta.
 */
#include "ops_internal.h"

#ifdef OPS_HAVE_AVX2

#include <immintrin.h>
#include <string.h>

#include "ops.h"
#include "quant.h"

#define TARGET __attribute__((target("avx2,fma,f16c")))

TARGET static inline float f16(uint16_t h) { return _cvtsh_ss(h); }

TARGET static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

TARGET static inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* Prodotto di due vettori di 32 int8 con segno -> 8 somme parziali int32.
 * maddubs vuole il primo operando senza segno: si usa |a| e si sposta il segno di a su b. */
TARGET static inline __m256i dot_i8(__m256i a, __m256i b) {
    __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(a, a), _mm256_sign_epi8(b, a));
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

/* 16 byte di nibble -> 32 valori 0..15: i 4 bit bassi nella metà bassa, gli alti nella metà alta
 * (è l'ordine degli elementi in Q4_0/Q5_0). */
TARGET static inline __m256i nibbles_32(const uint8_t *qs) {
    __m128i t = _mm_loadu_si128((const __m128i *)qs);
    __m256i v = _mm256_inserti128_si256(_mm256_castsi128_si256(t), _mm_srli_epi16(t, 4), 1);
    return _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
}

/* 32 bit -> 32 byte: 0xFF dove il bit è 1. Ogni byte di uscita prende il byte della parola che
 * contiene il suo bit, poi si confronta con una maschera che ha a 0 solo quel bit. */
TARGET static inline __m256i bits_to_bytes(uint32_t bits) {
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), shuf);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7FBFDFEFF7FBFDFE));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

/* ---------------------------------------------------------------- blocchi da 32 (attivazioni Q8_0) */

/* Pesi in 32 byte SENZA segno (u) più uno spostamento costante (bias): valore = u − bias.
 *   Q8_0: u = q (con segno, bias 0: si usa il trucco del segno);  Q4_0: u = 0..15, bias 8;
 *   Q5_0: u = 0..31, bias 16.
 * Con pesi senza segno maddubs si usa direttamente, e Σ (u − bias)·y = Σ u·y − bias·Σy, dove Σy
 * (somma delle attivazioni del blocco) si calcola una volta per matmul. */
TARGET static inline __m256i decode_q8_0(const block_q8_0 *b) { return _mm256_loadu_si256((const __m256i *)b->qs); }
TARGET static inline __m256i decode_q4_0(const block_q4_0 *b) { return nibbles_32(b->qs); }
TARGET static inline __m256i decode_q5_0(const block_q5_0 *b) {
    uint32_t qh;
    memcpy(&qh, b->qh, 4);
    return _mm256_or_si256(nibbles_32(b->qs), _mm256_and_si256(bits_to_bytes(qh), _mm256_set1_epi8(0x10)));
}
enum { BIAS_q8_0 = 0, BIAS_q4_0 = 8, BIAS_q5_0 = 16 };

/* Σ u·y per un blocco, come 8 float. Q8_0 (con segno) passa per |u| e il segno spostato su y. */
TARGET static inline __m256 block_dot_signed(__m256i au, __m256i u, const block_q8_0 *y) {
    __m256i p16 = _mm256_maddubs_epi16(au, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)y->qs), u));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
}
TARGET static inline __m256 block_dot_unsigned(__m256i u, const block_q8_0 *y) {
    __m256i p16 = _mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)y->qs));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
}

/* Scale (in float) e somme dei blocchi di attivazioni: calcolate una volta per matmul, servono a
 * tutte le righe dei pesi. DX_MAX limita lo spazio sullo stack (64 KB per scale e somme). */
#define DX_MAX 8192

static void activation_info(const block_q8_0 *x, int n_blocks, float *dx, float *sx) {
    for (int i = 0; i < n_blocks; i++) {
        dx[i] = op_f16_to_f32(x[i].d);
        int s = 0;
        for (int k = 0; k < 32; k++) s += x[i].qs[k];
        sx[i] = dx[i] * (float)s; /* Σy in unità reali: la correzione è −bias·d_w·sx */
    }
}

/* Per ogni tipo:
 * - dot: un token, due blocchi per iterazione su due accumulatori (latenze sovrapposte);
 * - matmul_rows: le scale delle attivazioni sono pre-convertite; con 4 token alla volta ogni blocco
 *   di pesi viene decodificato una volta sola e usato per i 4 token (prefill). */
#define BLOCK_TERM(S, U, AU, Y)                                                                       \
    (BIAS_##S ? block_dot_unsigned(U, Y) : block_dot_signed(AU, U, Y))

#define DEFINE_Q0_KERNELS(S, BLOCK)                                                                     \
    TARGET static float dot_rows_##S(const BLOCK *w, const block_q8_0 *x, const float *dx,            \
                                     const float *sx, int nb) {                                        \
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;                                                \
        float corr = 0.0f;                                                                             \
        int i = 0;                                                                                     \
        for (; i + 2 <= nb; i += 2) {                                                                  \
            __m256i u0 = decode_##S(&w[i]), u1 = decode_##S(&w[i + 1]);                               \
            float d0 = f16(w[i].d), d1 = f16(w[i + 1].d);                                              \
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * dx[i]),                                         \
                                   BLOCK_TERM(S, u0, _mm256_sign_epi8(u0, u0), &x[i]), acc0);          \
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1 * dx[i + 1]),                                     \
                                   BLOCK_TERM(S, u1, _mm256_sign_epi8(u1, u1), &x[i + 1]), acc1);      \
            if (BIAS_##S) corr += d0 * sx[i] + d1 * sx[i + 1];                                        \
        }                                                                                              \
        for (; i < nb; i++) {                                                                          \
            __m256i u0 = decode_##S(&w[i]);                                                            \
            float d0 = f16(w[i].d);                                                                    \
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * dx[i]),                                         \
                                   BLOCK_TERM(S, u0, _mm256_sign_epi8(u0, u0), &x[i]), acc0);          \
            if (BIAS_##S) corr += d0 * sx[i];                                                          \
        }                                                                                              \
        return hsum_ps(_mm256_add_ps(acc0, acc1)) - BIAS_##S * corr;                                   \
    }                                                                                                  \
                                                                                                       \
    TARGET float quant_dot_##S##_avx2(const void *wv, const void *xv, int n) {                         \
        int nb = n / 32;                                                                               \
        float dx[DX_MAX / 4], sx[DX_MAX / 4];                                                           \
        if (nb > DX_MAX / 4) { /* righe enormi: a pezzi */                                             \
            float s = 0.0f;                                                                            \
            for (int i = 0; i < nb; i += DX_MAX / 4) {                                                 \
                int m = nb - i < DX_MAX / 4 ? nb - i : DX_MAX / 4;                                     \
                s += quant_dot_##S##_avx2((const BLOCK *)wv + i, (const block_q8_0 *)xv + i, m * 32);  \
            }                                                                                          \
            return s;                                                                                  \
        }                                                                                              \
        activation_info(xv, nb, dx, sx);                                                               \
        return dot_rows_##S(wv, xv, dx, sx, nb);                                                       \
    }                                                                                                  \
                                                                                                       \
    TARGET void quant_matmul_rows_##S##_avx2(float *out, const void *xq, const void *wv, int n_tok,    \
                                             int n_in, int n_out, int row0, int row1) {                 \
        int nb = n_in / 32;                                                                            \
        const block_q8_0 *x = xq;                                                                      \
        float dx[DX_MAX], sx[DX_MAX];                                                                  \
        if ((size_t)n_tok * nb > DX_MAX) { /* blocco di token troppo grande: a gruppi */               \
            int step = DX_MAX / nb < 4 ? 1 : DX_MAX / nb / 4 * 4;                                      \
            if (nb > DX_MAX) step = 0;                                                                 \
            if (step == 0) {                                                                           \
                for (int j = row0; j < row1; j++)                                                      \
                    for (int t = 0; t < n_tok; t++)                                                    \
                        out[(size_t)t * n_out + j] = quant_dot_##S##_avx2((const BLOCK *)wv + (size_t)j * nb, \
                                                                         x + (size_t)t * nb, n_in);    \
                return;                                                                                \
            }                                                                                          \
            for (int t = 0; t < n_tok; t += step) {                                                    \
                int m = n_tok - t < step ? n_tok - t : step;                                           \
                quant_matmul_rows_##S##_avx2(out + (size_t)t * n_out, x + (size_t)t * nb, wv, m, n_in, n_out, \
                                             row0, row1);                                              \
            }                                                                                          \
            return;                                                                                    \
        }                                                                                              \
        activation_info(x, n_tok * nb, dx, sx);                                                        \
        for (int j = row0; j < row1; j++) {                                                            \
            const BLOCK *w = (const BLOCK *)wv + (size_t)j * nb;                                       \
            int t = 0;                                                                                 \
            for (; t + 4 <= n_tok; t += 4) {                                                           \
                const block_q8_0 *x0 = x + (size_t)t * nb, *x1 = x0 + nb, *x2 = x1 + nb, *x3 = x2 + nb; \
                const float *d0 = dx + (size_t)t * nb, *d1 = d0 + nb, *d2 = d1 + nb, *d3 = d2 + nb;    \
                const float *s0 = sx + (size_t)t * nb, *s1 = s0 + nb, *s2 = s1 + nb, *s3 = s2 + nb;    \
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;                            \
                float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;                                      \
                for (int i = 0; i < nb; i++) {                                                         \
                    __m256i u = decode_##S(&w[i]), au = _mm256_sign_epi8(u, u);                        \
                    float dw = f16(w[i].d);                                                            \
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(dw * d0[i]), BLOCK_TERM(S, u, au, &x0[i]), a0); \
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(dw * d1[i]), BLOCK_TERM(S, u, au, &x1[i]), a1); \
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(dw * d2[i]), BLOCK_TERM(S, u, au, &x2[i]), a2); \
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(dw * d3[i]), BLOCK_TERM(S, u, au, &x3[i]), a3); \
                    if (BIAS_##S) {                                                                    \
                        c0 += dw * s0[i];                                                              \
                        c1 += dw * s1[i];                                                              \
                        c2 += dw * s2[i];                                                              \
                        c3 += dw * s3[i];                                                              \
                    }                                                                                  \
                }                                                                                      \
                out[(size_t)t * n_out + j] = hsum_ps(a0) - BIAS_##S * c0;                              \
                out[(size_t)(t + 1) * n_out + j] = hsum_ps(a1) - BIAS_##S * c1;                        \
                out[(size_t)(t + 2) * n_out + j] = hsum_ps(a2) - BIAS_##S * c2;                        \
                out[(size_t)(t + 3) * n_out + j] = hsum_ps(a3) - BIAS_##S * c3;                        \
            }                                                                                          \
            for (; t < n_tok; t++)                                                                     \
                out[(size_t)t * n_out + j] = dot_rows_##S(w, x + (size_t)t * nb, dx + (size_t)t * nb,  \
                                                          sx + (size_t)t * nb, nb);                    \
        }                                                                                              \
    }

DEFINE_Q0_KERNELS(q8_0, block_q8_0)
DEFINE_Q0_KERNELS(q4_0, block_q4_0)
DEFINE_Q0_KERNELS(q5_0, block_q5_0)

/* ---------------------------------------------------------------- K-quant (attivazioni Q8_K) */

TARGET float quant_dot_q4_K_avx2(const void *w, const void *x, int n) {
    const block_q4_K *a = w;
    const block_q8_K *b = x;
    const __m256i low4 = _mm256_set1_epi8(0x0F);
    float acc = 0.0f;
    for (int i = 0; i < n / 256; i++) {
        uint8_t sc[8], m[8];
        for (int j = 0; j < 8; j++) q4k_scale_min(j, a[i].scales, &sc[j], &m[j]);
        int32_t msum = 0;
        for (int j = 0; j < 8; j++) msum += m[j] * (b[i].bsums[2 * j] + b[i].bsums[2 * j + 1]);

        __m256i isum = _mm256_setzero_si256();
        const uint8_t *q = a[i].qs;
        const int8_t *y = b[i].qs;
        for (int j = 0; j < 4; j++) { /* 32 byte di pesi = 64 elementi, due sotto-blocchi */
            __m256i q4 = _mm256_loadu_si256((const __m256i *)q);
            __m256i lo = _mm256_and_si256(q4, low4);                        /* 0..15, senza segno */
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), low4);
            __m256i p_lo = _mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)y));
            __m256i p_hi = _mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(y + 32)));
            /* madd con la scala del sotto-blocco: somma a coppie e moltiplica in un colpo. */
            isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p_lo, _mm256_set1_epi16(sc[2 * j])));
            isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p_hi, _mm256_set1_epi16(sc[2 * j + 1])));
            q += 32;
            y += 64;
        }
        acc = q4k_combine(acc, b[i].d, f16(a[i].d), f16(a[i].dmin), hsum_epi32(isum), msum);
    }
    return acc;
}

TARGET float quant_dot_q6_K_avx2(const void *w, const void *x, int n) {
    const block_q6_K *a = w;
    const block_q8_K *b = x;
    const __m256i low4 = _mm256_set1_epi8(0x0F), two = _mm256_set1_epi8(3);
    float acc = 0.0f;
    for (int i = 0; i < n / 256; i++) {
        /* Σ sc·(q − 32)·y = Σ sc·q·y − 32·Σ sc·Σy: q è 0..63 senza segno (va bene per maddubs)
         * e la correzione usa le somme bsums delle attivazioni. */
        int32_t corr = 0;
        for (int g = 0; g < 16; g++) corr += a[i].scales[g] * b[i].bsums[g];

        __m256i isum = _mm256_setzero_si256();
        const uint8_t *ql = a[i].ql, *qh = a[i].qh;
        const int8_t *y = b[i].qs;
        const int8_t *sc = a[i].scales;
        for (int half = 0; half < 2; half++) { /* 128 elementi: 64 byte di ql, 32 di qh */
            __m256i l0 = _mm256_loadu_si256((const __m256i *)ql);
            __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i h = _mm256_loadu_si256((const __m256i *)qh);
            __m256i q[4] = {
                _mm256_or_si256(_mm256_and_si256(l0, low4), _mm256_slli_epi16(_mm256_and_si256(h, two), 4)),
                _mm256_or_si256(_mm256_and_si256(l1, low4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), two), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), two), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), two), 4)),
            };
            for (int k = 0; k < 4; k++) { /* 32 elementi = 2 gruppi da 16 con scale diverse */
                __m256i p = _mm256_maddubs_epi16(q[k], _mm256_loadu_si256((const __m256i *)(y + 32 * k)));
                __m256i s = _mm256_inserti128_si256(_mm256_set1_epi16(sc[2 * k]), _mm_set1_epi16(sc[2 * k + 1]), 1);
                isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, s));
            }
            ql += 64;
            qh += 32;
            y += 128;
            sc += 8;
        }
        acc = q6k_combine(acc, b[i].d, f16(a[i].d), hsum_epi32(isum) - 32 * corr);
    }
    return acc;
}

/* ---------------------------------------------------------------- K-quant, 4 token per super-blocco
 *
 * Prefill: scale, minimi e valori a 4/6 bit di un super-blocco si decodificano una volta e si
 * moltiplicano per le attivazioni di 4 token. La parte intera di ogni super-blocco è la stessa del
 * prodotto scalare a un token (e la combinazione finale usa q4k_combine/q6k_combine): il risultato
 * è identico bit per bit a quello calcolato un token alla volta. */

TARGET void quant_matmul_rows_q4_K_avx2(float *out, const void *xq, const void *wv, int n_tok, int n_in,
                                        int n_out, int row0, int row1) {
    const int nb = n_in / 256;
    const block_q8_K *x = xq;
    const __m256i low4 = _mm256_set1_epi8(0x0F);
    for (int j = row0; j < row1; j++) {
        const block_q4_K *w = (const block_q4_K *)wv + (size_t)j * nb;
        int t = 0;
        for (; t + 4 <= n_tok; t += 4) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < nb; i++) {
                uint8_t sc[8], m[8];
                for (int k = 0; k < 8; k++) q4k_scale_min(k, w[i].scales, &sc[k], &m[k]);
                __m256i isum[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                                   _mm256_setzero_si256()};
                const uint8_t *q = w[i].qs;
                for (int g = 0; g < 4; g++) {
                    __m256i q4 = _mm256_loadu_si256((const __m256i *)q);
                    __m256i lo = _mm256_and_si256(q4, low4), hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), low4);
                    __m256i s_lo = _mm256_set1_epi16(sc[2 * g]), s_hi = _mm256_set1_epi16(sc[2 * g + 1]);
                    for (int tt = 0; tt < 4; tt++) {
                        const int8_t *y = x[(size_t)(t + tt) * nb + i].qs + 64 * g;
                        __m256i p_lo = _mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)y));
                        __m256i p_hi = _mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(y + 32)));
                        isum[tt] = _mm256_add_epi32(isum[tt], _mm256_madd_epi16(p_lo, s_lo));
                        isum[tt] = _mm256_add_epi32(isum[tt], _mm256_madd_epi16(p_hi, s_hi));
                    }
                    q += 32;
                }
                float d = f16(w[i].d), dmin = f16(w[i].dmin);
                for (int tt = 0; tt < 4; tt++) {
                    const block_q8_K *b = &x[(size_t)(t + tt) * nb + i];
                    int32_t msum = 0;
                    for (int k = 0; k < 8; k++) msum += m[k] * (b->bsums[2 * k] + b->bsums[2 * k + 1]);
                    acc[tt] = q4k_combine(acc[tt], b->d, d, dmin, hsum_epi32(isum[tt]), msum);
                }
            }
            for (int tt = 0; tt < 4; tt++) out[(size_t)(t + tt) * n_out + j] = acc[tt];
        }
        for (; t < n_tok; t++) out[(size_t)t * n_out + j] = quant_dot_q4_K_avx2(w, x + (size_t)t * nb, n_in);
    }
}

TARGET void quant_matmul_rows_q6_K_avx2(float *out, const void *xq, const void *wv, int n_tok, int n_in,
                                        int n_out, int row0, int row1) {
    const int nb = n_in / 256;
    const block_q8_K *x = xq;
    const __m256i low4 = _mm256_set1_epi8(0x0F), two = _mm256_set1_epi8(3);
    for (int j = row0; j < row1; j++) {
        const block_q6_K *w = (const block_q6_K *)wv + (size_t)j * nb;
        int t = 0;
        for (; t + 4 <= n_tok; t += 4) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < nb; i++) {
                __m256i isum[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                                   _mm256_setzero_si256()};
                const uint8_t *ql = w[i].ql, *qh = w[i].qh;
                const int8_t *sc = w[i].scales;
                for (int half = 0; half < 2; half++) {
                    __m256i l0 = _mm256_loadu_si256((const __m256i *)ql);
                    __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                    __m256i h = _mm256_loadu_si256((const __m256i *)qh);
                    __m256i q[4] = {
                        _mm256_or_si256(_mm256_and_si256(l0, low4), _mm256_slli_epi16(_mm256_and_si256(h, two), 4)),
                        _mm256_or_si256(_mm256_and_si256(l1, low4),
                                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), two), 4)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4),
                                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), two), 4)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4),
                                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), two), 4)),
                    };
                    for (int k = 0; k < 4; k++) {
                        __m256i s = _mm256_inserti128_si256(_mm256_set1_epi16(sc[2 * k]), _mm_set1_epi16(sc[2 * k + 1]), 1);
                        for (int tt = 0; tt < 4; tt++) {
                            const int8_t *y = x[(size_t)(t + tt) * nb + i].qs + 128 * half + 32 * k;
                            __m256i p = _mm256_maddubs_epi16(q[k], _mm256_loadu_si256((const __m256i *)y));
                            isum[tt] = _mm256_add_epi32(isum[tt], _mm256_madd_epi16(p, s));
                        }
                    }
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
                float d = f16(w[i].d);
                for (int tt = 0; tt < 4; tt++) {
                    const block_q8_K *b = &x[(size_t)(t + tt) * nb + i];
                    int32_t corr = 0;
                    for (int g = 0; g < 16; g++) corr += w[i].scales[g] * b->bsums[g];
                    acc[tt] = q6k_combine(acc[tt], b->d, d, hsum_epi32(isum[tt]) - 32 * corr);
                }
            }
            for (int tt = 0; tt < 4; tt++) out[(size_t)(t + tt) * n_out + j] = acc[tt];
        }
        for (; t < n_tok; t++) out[(size_t)t * n_out + j] = quant_dot_q6_K_avx2(w, x + (size_t)t * nb, n_in);
    }
}

#endif

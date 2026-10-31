#ifndef OPS_H
#define OPS_H

/* Kernel in fp32. Ogni funzione ha una controparte NumPy in sardyne/ref/ops.py e un test di
 * confronto in tests/test_runtime.py.
 *
 * Convenzione dei pesi: matrice [n_out][n_in] row-major (layout di PyTorch e GGUF), quindi
 * out = x · Wᵀ. I pesi possono essere F32, F16 o BF16: vengono convertiti al volo dentro il
 * prodotto scalare, e l'accumulo avviene sempre in fp32.
 *
 * Due implementazioni, scelte a runtime alla prima chiamata di op_init():
 * - AVX2 + FMA + F16C (x86 Haswell e successivi), in ops_avx2.c;
 * - scalare portabile, con 4 accumulatori indipendenti.
 * ENGINE_NO_SIMD=1 nell'ambiente, oppure op_set_simd(0), forza la versione scalare.
 */

#include <stddef.h>
#include <stdint.h>

/* Tipi ggml supportati dal runtime (stessi id del formato GGUF). Q8_K serve solo alle attivazioni. */
enum { OP_F32 = 0, OP_F16 = 1, OP_Q4_0 = 2, OP_Q5_0 = 6, OP_Q8_0 = 8, OP_Q4_K = 12, OP_Q6_K = 14,
       OP_Q8_K = 15, OP_BF16 = 30 };

void op_init(void); /* tabella f16 e scelta dei kernel; idempotente e thread-safe */
int op_simd(void);  /* 1 se sono attivi i kernel AVX2 */
int op_set_simd(int enable); /* 0: scalare; 1: AVX2 se disponibile. Restituisce op_simd(). */
size_t op_type_size(int type); /* byte per elemento dei tipi non quantizzati, altrimenti 0 */
int op_type_supported(int type);  /* tipo di pesi accettato dal runtime */
int op_block_size(int type);      /* elementi per blocco (1 per F32/F16/BF16), 0 se sconosciuto */
size_t op_row_size(int type, int n); /* byte di una riga di n elementi (n multiplo del blocco) */
/* Tipo delle attivazioni per i pesi wtype: OP_F32 (attivazioni in float), OP_Q8_0 o OP_Q8_K. */
int op_vec_dot_type(int wtype);

float op_f16_to_f32(uint16_t h);
uint16_t op_f32_to_f16(float f); /* arrotondamento al pari, come NumPy e F16C */
void op_f16_to_f32_n(const uint16_t *x, float *out, size_t n);
void op_row_to_f32(const void *row, int type, float *out, size_t n);

/* out[t][j] = sum_i x[t][i] * w[j][i]   per t < n_tok e row0 <= j < row1 (out ha n_out colonne).
 * Con più thread ognuno calcola un intervallo di righe: il risultato di una riga non dipende
 * da come le righe sono divise, purché row0 sia pari. */
void op_matmul_rows(float *out, const float *x, const void *w, int wtype, int n_tok, int n_in, int n_out,
                    int row0, int row1);
/* Con pesi quantizzati quantizza prima le attivazioni (in una memoria temporanea: per i test). */
void op_matmul(float *out, const float *x, const void *w, int wtype, int n_tok, int n_in, int n_out);

/* Pesi quantizzati. Le attivazioni si quantizzano una volta (vtype = op_vec_dot_type(wtype)),
 * per le righe di token t0..t1, poi le righe dei pesi si possono dividere tra i thread. */
void op_quantize_rows(int vtype, const float *x, void *xq, int n_in, int t0, int t1);
void op_matmul_q_rows(float *out, const void *xq, const void *w, int wtype, int n_tok, int n_in, int n_out,
                      int row0, int row1);
float op_vec_dot_q(int wtype, const void *w, const void *xq, int n);
void op_add_bias(float *x, const float *b, int n_tok, int n);

float op_dot_f32(const float *a, const float *b, int n);
void op_axpy(float *y, float a, const float *x, int n); /* y += a * x */

void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void op_softmax(float *x, int n);
void op_silu_mul(float *gate, const float *up, int n); /* gate = silu(gate) * up */

/* Tabelle cos/sin della RoPE: [n_pos][head_dim/2], angolo = pos * theta^(-2i/head_dim). */
void op_rope_table(float *cos_out, float *sin_out, int n_pos, int head_dim, float theta);
/* Ruota in place n_heads vettori consecutivi di lunghezza head_dim alla posizione di cos/sin
 * (righe della tabella). interleaved: coppie (2i, 2i+1); altrimenti coppie (i, i + head_dim/2). */
void op_rope(float *x, int n_heads, int head_dim, const float *cos, const float *sin, int interleaved);

/* Attenzione di una testa su n_kv posizioni: k, v [n_kv][head_dim]; scores: lavoro [n_kv]. */
void op_attention_head(float *out, const float *q, const float *k, const float *v, int head_dim, int n_kv,
                       float *scores);
/* Tutte le teste di un token, con GQA. q, out: [n_head][head_dim];
 * k_cache, v_cache: [n_kv_head][stride][head_dim]; la testa h usa la testa KV h / (n_head / n_kv_head). */
void op_attention(float *out, const float *q, const float *k_cache, const float *v_cache,
                  int n_head, int n_kv_head, int head_dim, int n_kv, int stride, float *scores);

#endif

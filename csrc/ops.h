#ifndef OPS_H
#define OPS_H

/* Kernel scalari in fp32 (Fase 3). Ogni funzione ha una controparte NumPy in engine/ref/ops.py
 * e un test di confronto in tests/test_runtime.py.
 *
 * Convenzione dei pesi: matrice [n_out][n_in] row-major (layout di PyTorch e GGUF), quindi
 * out = x · Wᵀ. I pesi possono essere F32, F16 o BF16: vengono convertiti al volo dentro il
 * prodotto scalare, e l'accumulo avviene sempre in fp32.
 */

#include <stddef.h>
#include <stdint.h>

/* Tipi ggml dei pesi supportati dal runtime (stessi id del formato GGUF). */
enum { OP_F32 = 0, OP_F16 = 1, OP_BF16 = 30 };

void op_init(void); /* prepara la tabella f16 -> f32; idempotente e thread-safe */
size_t op_type_size(int type); /* byte per elemento, 0 se il tipo non è supportato */

float op_f16_to_f32(uint16_t h);
void op_f16_to_f32_n(const uint16_t *x, float *out, size_t n);
void op_row_to_f32(const void *row, int type, float *out, size_t n);

/* out[t][j] = sum_i x[t][i] * w[j][i]   per t < n_tok, j < n_out */
void op_matmul(float *out, const float *x, const void *w, int wtype, int n_tok, int n_in, int n_out);
void op_add_bias(float *x, const float *b, int n_tok, int n);

void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void op_softmax(float *x, int n);
void op_silu_mul(float *gate, const float *up, int n); /* gate = silu(gate) * up */

/* Tabelle cos/sin della RoPE: [n_pos][head_dim/2], angolo = pos * theta^(-2i/head_dim). */
void op_rope_table(float *cos_out, float *sin_out, int n_pos, int head_dim, float theta);
/* Ruota in place n_heads vettori consecutivi di lunghezza head_dim alla posizione di cos/sin
 * (righe della tabella). interleaved: coppie (2i, 2i+1); altrimenti coppie (i, i + head_dim/2). */
void op_rope(float *x, int n_heads, int head_dim, const float *cos, const float *sin, int interleaved);

/* Attenzione di un token su n_kv posizioni della cache, con GQA.
 * q, out: [n_head][head_dim]; k_cache, v_cache: [n_kv_head][stride][head_dim];
 * la testa h usa la testa KV h / (n_head / n_kv_head). scores: spazio di lavoro [n_kv]. */
void op_attention(float *out, const float *q, const float *k_cache, const float *v_cache,
                  int n_head, int n_kv_head, int head_dim, int n_kv, int stride, float *scores);

#endif

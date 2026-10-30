#ifndef ENGINE_H
#define ENGINE_H

/* API pubblica del runtime C.
 *
 *   engine_model  pesi del modello, letti da un file GGUF tramite mmap (sola lettura, condivisibili)
 *   engine_ctx    stato di una sessione: KV cache e memoria di lavoro, allocate una volta sola
 *
 * Uso tipico:
 *   engine_model *m = engine_model_load("model.gguf", err, sizeof err);
 *   engine_ctx *c = engine_ctx_new(m, 2048, 64, 0, 0, err, sizeof err);
 *   engine_eval(c, prompt, n_prompt);            // prefill
 *   const float *logits = engine_logits(c);      // [n_vocab] dell'ultimo token
 *   ... scelta del token successivo, poi engine_eval(c, &tok, 1) ...   // decode
 *   engine_ctx_free(c); engine_model_free(m);
 */

#include <stddef.h>
#include <stdint.h>

typedef struct engine_model engine_model;
typedef struct engine_ctx engine_ctx;

typedef struct {
    int32_t n_vocab;
    int32_t dim;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_kv_head;
    int32_t head_dim;
    int32_t ffn_dim;
    int32_t max_pos;
    float norm_eps;
    float rope_theta;
    int32_t rope_interleaved; /* 1: coppie (2i, 2i+1) come "llama"; 0: coppie (i, i+hd/2) come "qwen2" */
    int32_t qkv_bias;
    int32_t tied_embeddings;
    int32_t qk_norm; /* Qwen3: RMSNorm per testa su q e k, prima della RoPE */
    int32_t n_expert, n_expert_used, ffn_exp_dim; /* MoE: esperti per strato (0 = MLP normale), scelti per token, dimensione interna */
} engine_model_info;

/* Fase 0: smoke test della toolchain Python <-> C. */
const char *engine_version(void);
void engine_add_f32(const float *a, const float *b, float *out, size_t n);

/* Carica un GGUF di architettura "llama", "qwen2", "qwen3" o "qwen3moe" con pesi F32/F16/BF16 o quantizzati (Q8_0, Q4_0,
 * Q5_0, Q4_K, Q6_K: i file Q8_0 e Q4_K_M più diffusi). NULL in caso di errore. */
engine_model *engine_model_load(const char *path, char *err, size_t err_len);
void engine_model_free(engine_model *m);
void engine_model_get_info(const engine_model *m, engine_model_info *out);

/* max_seq: lunghezza massima del contesto (KV cache); n_batch: token elaborati insieme in un
 * passo; logits_all: 1 per avere i logits di ogni token (allora engine_eval accetta al massimo
 * n_batch token per chiamata), 0 per avere solo quelli dell'ultimo; n_threads: thread di calcolo
 * (0 = uno per core). Il risultato non dipende dal numero di thread: è identico bit per bit. */
engine_ctx *engine_ctx_new(const engine_model *m, int32_t max_seq, int32_t n_batch, int32_t logits_all,
                           int32_t n_threads, char *err, size_t err_len);
int32_t engine_ctx_n_threads(const engine_ctx *c);
void engine_ctx_free(engine_ctx *c);

/* Elabora n token che proseguono quelli già visti. 0 se ok, -1 in caso di errore (contesto
 * pieno, token fuori dal vocabolario...): il motivo è in engine_last_error e il contesto resta
 * com'era prima della chiamata. */
int engine_eval(engine_ctx *c, const int32_t *tokens, int32_t n);
/* Logits dell'ultima engine_eval: [n_logits][n_vocab], valido fino alla chiamata successiva. */
const float *engine_logits(const engine_ctx *c);
int32_t engine_n_logits(const engine_ctx *c);
int32_t engine_n_past(const engine_ctx *c); /* token presenti nella KV cache */
void engine_reset(engine_ctx *c);           /* svuota il contesto */
/* Torna alla posizione n <= n_past: i primi n token restano nella KV cache. 0, oppure -1 se n non è valido. */
int engine_truncate(engine_ctx *c, int32_t n);
const char *engine_last_error(const engine_ctx *c);

#endif

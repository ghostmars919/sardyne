/* Runtime fp32 per modelli Llama-like (architetture GGUF "llama", "qwen2" e "qwen3").
 *
 * È la traduzione in C di engine/ref/llama.py: stesse operazioni, nello stesso ordine.
 * Le matrici dei pesi restano nella mmap del file GGUF; norm e bias (piccoli) vengono
 * convertiti in fp32 al caricamento.
 */
#define _POSIX_C_SOURCE 200809L

#include "engine.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "ops.h"
#include "threadpool.h"

#define MAX_SEQ_LIMIT (1 << 20)
#define MAX_BATCH_LIMIT 4096
#define ALIGN 64

typedef struct {
    const void *data; /* dentro la mmap */
    int type;         /* OP_F32, OP_F16, OP_BF16 o un tipo quantizzato (OP_Q8_0, OP_Q4_K, ...) */
} matrix;

typedef struct {
    float *attn_norm, *ffn_norm;
    float *bq, *bk, *bv; /* NULL senza bias */
    float *q_norm, *k_norm; /* NULL senza qk-norm */
    matrix wq, wk, wv, wo, gate, up, down;
} layer;

struct engine_model {
    gguf_file *gf;
    engine_model_info info;
    matrix tok_embd, output;
    float *output_norm;
    layer *layers;
    float *vectors; /* un'unica allocazione per tutti i vettori fp32 */
};

struct engine_ctx {
    const engine_model *m;
    int max_seq, n_batch, logits_all, n_past, n_logits;
    int n_threads;
    threadpool *pool;
    void *arena;
    float *k_cache, *v_cache; /* [n_layer][n_kv_head][max_seq][head_dim] */
    float *rope_cos, *rope_sin; /* [max_seq][head_dim/2] */
    float *x, *xb, *q, *k, *v, *att, *hb, *hb2, *scores, *logits;
    /* Attivazioni quantizzate per i pesi quantizzati: [0] in Q8_0, [1] in Q8_K. Si quantizzano
     * una volta e servono a tutte le matmul con lo stesso ingresso (q, k, v; gate e up). */
    void *xq[2];
    const float *xq_src[2];
    int xq_n_tok[2], xq_n_in[2];
    char err[256];
};

/* ---------------------------------------------------------------- errori */

static void set_err(char *err, size_t len, const char *fmt, ...) {
    if (!err || !len) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, len, fmt, ap);
    va_end(ap);
}

/* ---------------------------------------------------------------- lettura dei metadati */

typedef struct {
    const gguf_file *gf;
    const char *arch;
    char *err;
    size_t err_len;
} meta;

static int64_t arch_key(const meta *mt, const char *suffix) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", mt->arch, suffix);
    return gguf_find_kv(mt->gf, key);
}

/* Intero obbligatorio (def < 0) o con valore predefinito; -1 in caso di errore. */
static int64_t meta_int(const meta *mt, const char *suffix, int64_t def) {
    int64_t i = arch_key(mt, suffix);
    uint64_t v;
    if (i < 0) {
        if (def < 0) set_err(mt->err, mt->err_len, "metadato mancante: %s.%s", mt->arch, suffix);
        return def;
    }
    if (gguf_kv_get_u64(mt->gf, i, &v) || v == 0 || v > (1u << 30)) {
        set_err(mt->err, mt->err_len, "metadato non valido: %s.%s", mt->arch, suffix);
        return -1;
    }
    return (int64_t)v;
}

static int meta_float(const meta *mt, const char *suffix, double def, float *out) {
    int64_t i = arch_key(mt, suffix);
    double v = def;
    if (i < 0 && def < 0) {
        set_err(mt->err, mt->err_len, "metadato mancante: %s.%s", mt->arch, suffix);
        return -1;
    }
    if (i >= 0 && gguf_kv_get_f64(mt->gf, i, &v)) {
        set_err(mt->err, mt->err_len, "metadato non valido: %s.%s", mt->arch, suffix);
        return -1;
    }
    if (!(v > 0.0) || v > 1e12) {
        set_err(mt->err, mt->err_len, "valore non valido per %s.%s", mt->arch, suffix);
        return -1;
    }
    *out = (float)v;
    return 0;
}

/* ---------------------------------------------------------------- tensori */

static const char *LAYER_SUFFIXES[] = {
    "attn_norm.weight", "attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias",
    "attn_v.weight", "attn_v.bias", "attn_q_norm.weight", "attn_k_norm.weight",
    "attn_output.weight", "ffn_norm.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
};

/* Come _GGUF_TENSOR in engine/ref/llama.py: un tensore sconosciuto indica una variante
 * dell'architettura che non sappiamo eseguire (es. rope_freqs.weight dei Llama 3.1). */
static int known_tensor(const char *name, int n_layer) {
    if (!strcmp(name, "token_embd.weight") || !strcmp(name, "output_norm.weight") || !strcmp(name, "output.weight"))
        return 1;
    if (strncmp(name, "blk.", 4) != 0) return 0;
    const char *p = name + 4;
    long idx = 0;
    if (*p < '0' || *p > '9') return 0;
    while (*p >= '0' && *p <= '9') {
        idx = idx * 10 + (*p - '0');
        if (idx >= n_layer) return 0;
        p++;
    }
    if (*p++ != '.') return 0;
    for (size_t k = 0; k < sizeof(LAYER_SUFFIXES) / sizeof(LAYER_SUFFIXES[0]); k++)
        if (!strcmp(p, LAYER_SUFFIXES[k])) return 1;
    return 0;
}

static const gguf_tensor *find(const gguf_file *gf, const char *name) {
    int64_t i = gguf_find_tensor(gf, name);
    return i < 0 ? NULL : gguf_get_tensor(gf, (uint64_t)i);
}

/* Matrice [n_out][n_in] (ne = [n_in, n_out]) di tipo supportato. */
static int get_matrix(const gguf_file *gf, const char *name, int n_in, int n_out, matrix *out,
                      char *err, size_t err_len) {
    const gguf_tensor *t = find(gf, name);
    if (!t) {
        set_err(err, err_len, "tensore mancante: %s", name);
        return -1;
    }
    if (t->n_dims != 2 || t->ne[0] != (uint64_t)n_in || t->ne[1] != (uint64_t)n_out) {
        set_err(err, err_len, "%s: dimensioni [%llu, %llu], attese [%d, %d]", name,
                (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], n_in, n_out);
        return -1;
    }
    if (!op_type_supported((int)t->type)) {
        set_err(err, err_len, "%s: tipo %s non supportato dal runtime (supportati: F32, F16, BF16, Q8_0, "
                "Q4_0, Q5_0, Q4_K, Q6_K)", name, gguf_type_name(t->type));
        return -1;
    }
    out->data = t->data;
    out->type = (int)t->type;
    return 0;
}

/* Vettore di n elementi convertito in fp32 in dst. */
static int get_vector(const gguf_file *gf, const char *name, int n, float *dst, char *err, size_t err_len) {
    const gguf_tensor *t = find(gf, name);
    if (!t) {
        set_err(err, err_len, "tensore mancante: %s", name);
        return -1;
    }
    if (t->n_dims != 1 || t->ne[0] != (uint64_t)n) {
        set_err(err, err_len, "%s: dimensione %llu, attesa %d", name, (unsigned long long)t->ne[0], n);
        return -1;
    }
    if (!op_type_supported((int)t->type)) {
        set_err(err, err_len, "%s: tipo %s non supportato", name, gguf_type_name(t->type));
        return -1;
    }
    op_row_to_f32(t->data, (int)t->type, dst, (size_t)n);
    return 0;
}

/* ---------------------------------------------------------------- caricamento */

void engine_model_free(engine_model *m) {
    if (!m) return;
    free(m->layers);
    free(m->vectors);
    gguf_close(m->gf);
    free(m);
}

void engine_model_get_info(const engine_model *m, engine_model_info *out) {
    if (m && out) *out = m->info;
}

static int read_config(engine_model *m, char *err, size_t err_len) {
    const gguf_file *gf = m->gf;
    uint64_t len;
    const char *arch = gguf_kv_get_str(gf, gguf_find_kv(gf, "general.architecture"), &len);
    char arch_buf[32];
    if (!arch || len == 0 || len >= sizeof(arch_buf)) {
        set_err(err, err_len, "general.architecture mancante o non valido");
        return -1;
    }
    memcpy(arch_buf, arch, len);
    arch_buf[len] = '\0';
    engine_model_info *in = &m->info;
    if (!strcmp(arch_buf, "llama")) in->rope_interleaved = 1;
    else if (!strcmp(arch_buf, "qwen2") || !strcmp(arch_buf, "qwen3")) in->rope_interleaved = 0;
    else {
        set_err(err, err_len, "architettura %s non supportata", arch_buf);
        return -1;
    }
    meta mt = {gf, arch_buf, err, err_len};

    const char *scaling = gguf_kv_get_str(gf, arch_key(&mt, "rope.scaling.type"), &len);
    if ((scaling && !(len == 4 && !memcmp(scaling, "none", 4))) || find(gf, "rope_freqs.weight")) {
        set_err(err, err_len, "rope scaling (anche 'llama3' via rope_freqs.weight) arriva in Fase 6");
        return -1;
    }

    int64_t v;
    if ((v = meta_int(&mt, "embedding_length", -1)) < 0) return -1;
    in->dim = (int32_t)v;
    if ((v = meta_int(&mt, "block_count", -1)) < 0) return -1;
    in->n_layer = (int32_t)v;
    if ((v = meta_int(&mt, "feed_forward_length", -1)) < 0) return -1;
    in->ffn_dim = (int32_t)v;
    if ((v = meta_int(&mt, "attention.head_count", -1)) < 0) return -1;
    in->n_head = (int32_t)v;
    if ((v = meta_int(&mt, "attention.head_count_kv", in->n_head)) < 0) return -1;
    in->n_kv_head = (int32_t)v;
    if ((v = meta_int(&mt, "attention.key_length", in->dim / in->n_head)) <= 0) {
        if (v == 0) set_err(err, err_len, "head_dim nullo");
        return -1;
    }
    in->head_dim = (int32_t)v;
    if (meta_int(&mt, "attention.value_length", in->head_dim) != in->head_dim) {
        set_err(err, err_len, "key_length diverso da value_length");
        return -1;
    }
    if (meta_int(&mt, "rope.dimension_count", in->head_dim) != in->head_dim) {
        set_err(err, err_len, "RoPE parziale (rope.dimension_count < head_dim) non supportata");
        return -1;
    }
    if ((v = meta_int(&mt, "context_length", 2048)) < 0) return -1;
    in->max_pos = (int32_t)v;
    if (meta_float(&mt, "attention.layer_norm_rms_epsilon", -1.0, &in->norm_eps)) return -1;
    if (meta_float(&mt, "rope.freq_base", 10000.0, &in->rope_theta)) return -1;
    if (in->n_head % in->n_kv_head || in->head_dim % 2) {
        set_err(err, err_len, "teste o head_dim non validi (n_head %% n_kv_head, head_dim pari)");
        return -1;
    }

    const gguf_tensor *emb = find(gf, "token_embd.weight");
    if (!emb || emb->n_dims != 2 || emb->ne[1] == 0 || emb->ne[1] > (1u << 30)) {
        set_err(err, err_len, "token_embd.weight mancante o non valido");
        return -1;
    }
    in->n_vocab = (int32_t)emb->ne[1];
    in->tied_embeddings = find(gf, "output.weight") == NULL;
    in->qkv_bias = find(gf, "blk.0.attn_q.bias") != NULL;
    in->qk_norm = find(gf, "blk.0.attn_q_norm.weight") != NULL;

    for (uint64_t i = 0; i < gguf_n_tensors(gf); i++) {
        const char *name = gguf_get_tensor(gf, i)->name;
        if (!known_tensor(name, in->n_layer)) {
            set_err(err, err_len, "tensore non supportato: %s", name);
            return -1;
        }
    }
    return 0;
}

engine_model *engine_model_load(const char *path, char *err, size_t err_len) {
    op_init();
    if (err && err_len) err[0] = '\0';
    engine_model *m = calloc(1, sizeof(*m));
    if (!m) {
        set_err(err, err_len, "memoria esaurita");
        return NULL;
    }
    if (!(m->gf = gguf_open(path, err, err_len)) || read_config(m, err, err_len)) goto fail;

    const engine_model_info *in = &m->info;
    int dim = in->dim, q_dim = in->n_head * in->head_dim, kv_dim = in->n_kv_head * in->head_dim;
    size_t per_layer = 2 * (size_t)dim + (in->qkv_bias ? (size_t)q_dim + 2 * (size_t)kv_dim : 0) +
                       (in->qk_norm ? 2 * (size_t)in->head_dim : 0);
    m->vectors = malloc(((size_t)dim + per_layer * in->n_layer) * sizeof(float));
    m->layers = calloc((size_t)in->n_layer, sizeof(layer));
    if (!m->vectors || !m->layers) {
        set_err(err, err_len, "memoria esaurita");
        goto fail;
    }

    float *vec = m->vectors;
    m->output_norm = vec;
    vec += dim;
    if (get_matrix(m->gf, "token_embd.weight", dim, in->n_vocab, &m->tok_embd, err, err_len) ||
        get_vector(m->gf, "output_norm.weight", dim, m->output_norm, err, err_len))
        goto fail;
    if (in->tied_embeddings) m->output = m->tok_embd;
    else if (get_matrix(m->gf, "output.weight", dim, in->n_vocab, &m->output, err, err_len)) goto fail;

    for (int l = 0; l < in->n_layer; l++) {
        layer *L = &m->layers[l];
        char name[64];
#define NAME(suffix) (snprintf(name, sizeof(name), "blk.%d." suffix, l), name)
        L->attn_norm = vec;
        vec += dim;
        L->ffn_norm = vec;
        vec += dim;
        if (get_vector(m->gf, NAME("attn_norm.weight"), dim, L->attn_norm, err, err_len) ||
            get_vector(m->gf, NAME("ffn_norm.weight"), dim, L->ffn_norm, err, err_len) ||
            get_matrix(m->gf, NAME("attn_q.weight"), dim, q_dim, &L->wq, err, err_len) ||
            get_matrix(m->gf, NAME("attn_k.weight"), dim, kv_dim, &L->wk, err, err_len) ||
            get_matrix(m->gf, NAME("attn_v.weight"), dim, kv_dim, &L->wv, err, err_len) ||
            get_matrix(m->gf, NAME("attn_output.weight"), q_dim, dim, &L->wo, err, err_len) ||
            get_matrix(m->gf, NAME("ffn_gate.weight"), dim, in->ffn_dim, &L->gate, err, err_len) ||
            get_matrix(m->gf, NAME("ffn_up.weight"), dim, in->ffn_dim, &L->up, err, err_len) ||
            get_matrix(m->gf, NAME("ffn_down.weight"), in->ffn_dim, dim, &L->down, err, err_len))
            goto fail;
        if (in->qkv_bias) {
            L->bq = vec;
            vec += q_dim;
            L->bk = vec;
            vec += kv_dim;
            L->bv = vec;
            vec += kv_dim;
            if (get_vector(m->gf, NAME("attn_q.bias"), q_dim, L->bq, err, err_len) ||
                get_vector(m->gf, NAME("attn_k.bias"), kv_dim, L->bk, err, err_len) ||
                get_vector(m->gf, NAME("attn_v.bias"), kv_dim, L->bv, err, err_len))
                goto fail;
        }
        if (in->qk_norm) {
            L->q_norm = vec;
            vec += in->head_dim;
            L->k_norm = vec;
            vec += in->head_dim;
            if (get_vector(m->gf, NAME("attn_q_norm.weight"), in->head_dim, L->q_norm, err, err_len) ||
                get_vector(m->gf, NAME("attn_k_norm.weight"), in->head_dim, L->k_norm, err, err_len))
                goto fail;
        }
#undef NAME
    }
    return m;

fail:
    engine_model_free(m);
    return NULL;
}

/* ---------------------------------------------------------------- contesto e arena */

/* Arena: un'unica allocazione per tutta la memoria di lavoro. Si fa un primo passaggio con
 * base == NULL che conta solo i byte, poi si alloca e si ripete l'assegnazione. */
typedef struct {
    uint8_t *base;
    size_t used;
    int overflow;
} arena;

static float *arena_floats(arena *a, size_t n) {
    if (n > (SIZE_MAX - a->used - ALIGN) / sizeof(float)) {
        a->overflow = 1;
        return NULL;
    }
    size_t off = (a->used + ALIGN - 1) / ALIGN * ALIGN;
    a->used = off + n * sizeof(float);
    return a->base ? (float *)(a->base + off) : NULL;
}

static void layout(engine_ctx *c, arena *a) {
    const engine_model_info *in = &c->m->info;
    size_t B = (size_t)c->n_batch, S = (size_t)c->max_seq;
    size_t kv = (size_t)in->n_layer * in->n_kv_head * S * in->head_dim;
    size_t q_dim = (size_t)in->n_head * in->head_dim, kv_dim = (size_t)in->n_kv_head * in->head_dim;
    c->k_cache = arena_floats(a, kv);
    c->v_cache = arena_floats(a, kv);
    c->rope_cos = arena_floats(a, S * in->head_dim / 2);
    c->rope_sin = arena_floats(a, S * in->head_dim / 2);
    c->x = arena_floats(a, B * in->dim);
    c->xb = arena_floats(a, B * in->dim);
    c->q = arena_floats(a, B * q_dim);
    c->k = arena_floats(a, B * kv_dim);
    c->v = arena_floats(a, B * kv_dim);
    c->att = arena_floats(a, B * q_dim);
    c->hb = arena_floats(a, B * in->ffn_dim);
    c->hb2 = arena_floats(a, B * in->ffn_dim);
    c->scores = arena_floats(a, S * (size_t)c->n_threads); /* uno spazio per thread */
    c->logits = arena_floats(a, (c->logits_all ? B : 1) * (size_t)in->n_vocab);
    /* Ingresso più lungo di una matmul, arrotondato ai blocchi da 32 e 256. */
    size_t max_in = (size_t)in->dim > q_dim ? (size_t)in->dim : q_dim;
    if ((size_t)in->ffn_dim > max_in) max_in = (size_t)in->ffn_dim;
    size_t row0 = (max_in + 31) / 32 * 34, rowk = (max_in + 255) / 256 * 292;
    c->xq[0] = arena_floats(a, (B * row0 + 3) / 4);
    c->xq[1] = arena_floats(a, (B * rowk + 3) / 4);
}

engine_ctx *engine_ctx_new(const engine_model *m, int32_t max_seq, int32_t n_batch, int32_t logits_all,
                           int32_t n_threads, char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!m) {
        set_err(err, err_len, "modello NULL");
        return NULL;
    }
    if (max_seq < 1 || max_seq > MAX_SEQ_LIMIT || max_seq > m->info.max_pos) {
        set_err(err, err_len, "max_seq=%d non valido (1..%d)", max_seq,
                m->info.max_pos < MAX_SEQ_LIMIT ? m->info.max_pos : MAX_SEQ_LIMIT);
        return NULL;
    }
    if (n_batch < 1 || n_batch > MAX_BATCH_LIMIT) {
        set_err(err, err_len, "n_batch=%d non valido (1..%d)", n_batch, MAX_BATCH_LIMIT);
        return NULL;
    }
    if (n_threads < 0 || n_threads > 256) {
        set_err(err, err_len, "n_threads=%d non valido (0..256)", n_threads);
        return NULL;
    }
    engine_ctx *c = calloc(1, sizeof(*c));
    if (!c) {
        set_err(err, err_len, "memoria esaurita");
        return NULL;
    }
    c->m = m;
    c->max_seq = max_seq;
    c->n_batch = n_batch;
    c->logits_all = logits_all != 0;
    c->n_threads = n_threads ? n_threads : tp_default_threads();
    if (!(c->pool = tp_new(c->n_threads))) {
        set_err(err, err_len, "impossibile creare %d thread", c->n_threads);
        free(c);
        return NULL;
    }

    arena a = {0};
    layout(c, &a); /* primo passaggio: solo conteggio */
    if (a.overflow) {
        set_err(err, err_len, "contesto troppo grande");
        tp_free(c->pool);
        free(c);
        return NULL;
    }
    size_t size = (a.used + ALIGN - 1) / ALIGN * ALIGN;
    if (posix_memalign(&c->arena, ALIGN, size) != 0) {
        set_err(err, err_len, "memoria esaurita (%.1f MB per contesto e KV cache)", size / 1e6);
        tp_free(c->pool);
        free(c);
        return NULL;
    }
    a = (arena){.base = c->arena};
    layout(c, &a);
    op_rope_table(c->rope_cos, c->rope_sin, max_seq, m->info.head_dim, m->info.rope_theta);
    return c;
}

void engine_ctx_free(engine_ctx *c) {
    if (!c) return;
    tp_free(c->pool);
    free(c->arena);
    free(c);
}

int32_t engine_ctx_n_threads(const engine_ctx *c) { return c ? c->n_threads : 0; }

/* ---------------------------------------------------------------- operazioni parallele
 *
 * Ogni operazione divide il lavoro in parti contigue, una per thread, e ogni elemento del
 * risultato è calcolato da un solo thread con la stessa sequenza di operazioni: per questo il
 * risultato è identico qualunque sia il numero di thread. */

static void split(int n, int ith, int nth, int *begin, int *end) {
    *begin = (int)((long long)n * ith / nth);
    *end = (int)((long long)n * (ith + 1) / nth);
}

typedef struct {
    float *out;
    const float *x;
    const matrix *w;
    int n_tok, n_in, n_out;
} matmul_job;

static void matmul_worker(void *arg, int ith, int nth) {
    const matmul_job *j = arg;
    int p0, p1; /* a coppie di righe: il micro-kernel lavora su 2 righe alla volta */
    split((j->n_out + 1) / 2, ith, nth, &p0, &p1);
    int r1 = 2 * p1 < j->n_out ? 2 * p1 : j->n_out;
    if (2 * p0 < r1) op_matmul_rows(j->out, j->x, j->w->data, j->w->type, j->n_tok, j->n_in, j->n_out, 2 * p0, r1);
}

typedef struct {
    float *out;
    const void *xq;
    const matrix *w;
    int n_tok, n_in, n_out;
} matmul_q_job;

static void matmul_q_worker(void *arg, int ith, int nth) {
    const matmul_q_job *j = arg;
    int r0, r1;
    split(j->n_out, ith, nth, &r0, &r1);
    if (r0 < r1) op_matmul_q_rows(j->out, j->xq, j->w->data, j->w->type, j->n_tok, j->n_in, j->n_out, r0, r1);
}

typedef struct {
    int vtype;
    const float *x;
    void *xq;
    int n_tok, n_in;
} quantize_job;

static void quantize_worker(void *arg, int ith, int nth) {
    const quantize_job *j = arg;
    int t0, t1;
    split(j->n_tok, ith, nth, &t0, &t1);
    if (t0 < t1) op_quantize_rows(j->vtype, j->x, j->xq, j->n_in, t0, t1);
}

/* out = x · Wᵀ. new_input = 1 quando x è cambiato dall'ultima chiamata: con pesi quantizzati le
 * attivazioni quantizzate si riusano finché l'ingresso è lo stesso (q, k, v; gate e up). */
static void matmul(engine_ctx *c, float *out, const float *x, const matrix *w, int n_tok, int n_in, int n_out,
                   int new_input) {
    if (new_input) c->xq_src[0] = c->xq_src[1] = NULL;
    int vt = op_vec_dot_type(w->type);
    if (vt == OP_F32) {
        matmul_job j = {out, x, w, n_tok, n_in, n_out};
        tp_run(c->pool, matmul_worker, &j);
        return;
    }
    int s = vt == OP_Q8_0 ? 0 : 1;
    if (c->xq_src[s] != x || c->xq_n_tok[s] != n_tok || c->xq_n_in[s] != n_in) {
        quantize_job qj = {vt, x, c->xq[s], n_tok, n_in};
        tp_run(c->pool, quantize_worker, &qj);
        c->xq_src[s] = x;
        c->xq_n_tok[s] = n_tok;
        c->xq_n_in[s] = n_in;
    }
    matmul_q_job j = {out, c->xq[s], w, n_tok, n_in, n_out};
    tp_run(c->pool, matmul_q_worker, &j);
}

typedef struct {
    engine_ctx *c;
    const float *kc, *vc;
    int n_tok, pos0;
} attention_job;

static void attention_worker(void *arg, int ith, int nth) {
    const attention_job *j = arg;
    const engine_model_info *in = &j->c->m->info;
    int hd = in->head_dim, q_dim = in->n_head * hd, n_rep = in->n_head / in->n_kv_head;
    size_t head_stride = (size_t)j->c->max_seq * hd;
    float *scores = j->c->scores + (size_t)ith * j->c->max_seq;
    int b, e; /* coppie (token, testa), indipendenti tra loro */
    split(j->n_tok * in->n_head, ith, nth, &b, &e);
    for (int task = b; task < e; task++) {
        int t = task / in->n_head, h = task % in->n_head;
        size_t kv = (size_t)(h / n_rep) * head_stride;
        /* Il token t vede le posizioni 0..pos0+t: è la maschera causale. */
        op_attention_head(j->c->att + (size_t)t * q_dim + (size_t)h * hd, j->c->q + (size_t)t * q_dim + (size_t)h * hd,
                          j->kc + kv, j->vc + kv, hd, j->pos0 + t + 1, scores);
    }
}

typedef struct {
    float *gate;
    const float *up;
    int n;
} silu_job;

static void silu_worker(void *arg, int ith, int nth) {
    const silu_job *j = arg;
    int b, e;
    split(j->n, ith, nth, &b, &e);
    op_silu_mul(j->gate + b, j->up + b, e - b);
}

/* ---------------------------------------------------------------- forward */

/* Elabora n <= n_batch token a partire dalla posizione n_past. Se out_rows > 0 calcola i logits
 * degli ultimi out_rows token. */
static void forward(engine_ctx *c, const int32_t *tok, int n, int out_rows) {
    const engine_model *m = c->m;
    const engine_model_info *in = &m->info;
    int dim = in->dim, hd = in->head_dim, half = hd / 2;
    int q_dim = in->n_head * hd, kv_dim = in->n_kv_head * hd;
    int pos0 = c->n_past;
    size_t head_stride = (size_t)c->max_seq * hd; /* tra due teste KV nella cache */
    size_t layer_stride = (size_t)in->n_kv_head * head_stride;
    size_t emb_row = op_row_size(m->tok_embd.type, dim);

    for (int t = 0; t < n; t++)
        op_row_to_f32((const uint8_t *)m->tok_embd.data + (size_t)tok[t] * emb_row, m->tok_embd.type,
                      c->x + (size_t)t * dim, (size_t)dim);

    for (int l = 0; l < in->n_layer; l++) {
        const layer *L = &m->layers[l];
        float *kc = c->k_cache + l * layer_stride, *vc = c->v_cache + l * layer_stride;

        /* --- attenzione: x = x + Wo · Attn(RoPE(Wq·norm(x)), RoPE(Wk·norm(x)), Wv·norm(x)) */
        for (int t = 0; t < n; t++)
            op_rmsnorm(c->xb + (size_t)t * dim, c->x + (size_t)t * dim, L->attn_norm, dim, in->norm_eps);
        matmul(c, c->q, c->xb, &L->wq, n, dim, q_dim, 1);
        matmul(c, c->k, c->xb, &L->wk, n, dim, kv_dim, 0);
        matmul(c, c->v, c->xb, &L->wv, n, dim, kv_dim, 0);
        if (L->bq) {
            op_add_bias(c->q, L->bq, n, q_dim);
            op_add_bias(c->k, L->bk, n, kv_dim);
            op_add_bias(c->v, L->bv, n, kv_dim);
        }
        for (int t = 0; t < n; t++) {
            int pos = pos0 + t;
            const float *cs = c->rope_cos + (size_t)pos * half, *sn = c->rope_sin + (size_t)pos * half;
            float *qt = c->q + (size_t)t * q_dim, *kt = c->k + (size_t)t * kv_dim, *vt = c->v + (size_t)t * kv_dim;
            if (L->q_norm) {
                for (int h = 0; h < in->n_head; h++) op_rmsnorm(qt + h * hd, qt + h * hd, L->q_norm, hd, in->norm_eps);
                for (int h = 0; h < in->n_kv_head; h++) op_rmsnorm(kt + h * hd, kt + h * hd, L->k_norm, hd, in->norm_eps);
            }
            op_rope(qt, in->n_head, hd, cs, sn, in->rope_interleaved);
            op_rope(kt, in->n_kv_head, hd, cs, sn, in->rope_interleaved);
            for (int h = 0; h < in->n_kv_head; h++) { /* K (già ruotato) e V nella cache */
                memcpy(kc + h * head_stride + (size_t)pos * hd, kt + (size_t)h * hd, (size_t)hd * sizeof(float));
                memcpy(vc + h * head_stride + (size_t)pos * hd, vt + (size_t)h * hd, (size_t)hd * sizeof(float));
            }
        }
        attention_job aj = {c, kc, vc, n, pos0};
        tp_run(c->pool, attention_worker, &aj);
        matmul(c, c->xb, c->att, &L->wo, n, q_dim, dim, 1);
        for (size_t i = 0; i < (size_t)n * dim; i++) c->x[i] += c->xb[i];

        /* --- MLP SwiGLU: x = x + down(silu(gate·norm(x)) * up·norm(x)) */
        for (int t = 0; t < n; t++)
            op_rmsnorm(c->xb + (size_t)t * dim, c->x + (size_t)t * dim, L->ffn_norm, dim, in->norm_eps);
        matmul(c, c->hb, c->xb, &L->gate, n, dim, in->ffn_dim, 1);
        matmul(c, c->hb2, c->xb, &L->up, n, dim, in->ffn_dim, 0);
        silu_job sj = {c->hb, c->hb2, n * in->ffn_dim};
        tp_run(c->pool, silu_worker, &sj);
        matmul(c, c->xb, c->hb, &L->down, n, in->ffn_dim, dim, 1);
        for (size_t i = 0; i < (size_t)n * dim; i++) c->x[i] += c->xb[i];
    }

    if (out_rows > 0) {
        /* Norm finale e proiezione sul vocabolario solo per le righe richieste:
         * con logits_all = 0 è una sola riga, la matmul più grande del passo. */
        const float *xs = c->x + (size_t)(n - out_rows) * dim;
        for (int r = 0; r < out_rows; r++)
            op_rmsnorm(c->xb + (size_t)r * dim, xs + (size_t)r * dim, m->output_norm, dim, in->norm_eps);
        matmul(c, c->logits, c->xb, &m->output, out_rows, dim, in->n_vocab, 1);
    }
    c->n_past += n;
}

int engine_eval(engine_ctx *c, const int32_t *tokens, int32_t n) {
    if (!c) return -1;
    c->err[0] = '\0';
    if (!tokens || n < 1) {
        snprintf(c->err, sizeof(c->err), "nessun token in ingresso");
        return -1;
    }
    if (c->logits_all && n > c->n_batch) {
        snprintf(c->err, sizeof(c->err), "con logits_all servono al massimo n_batch=%d token per chiamata", c->n_batch);
        return -1;
    }
    if (n > c->max_seq - c->n_past) {
        snprintf(c->err, sizeof(c->err), "contesto pieno: %d + %d > max_seq=%d", c->n_past, n, c->max_seq);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= c->m->info.n_vocab) {
            snprintf(c->err, sizeof(c->err), "token %d fuori dal vocabolario [0, %d)", tokens[i], c->m->info.n_vocab);
            return -1;
        }
    }
    /* Blocchi da n_batch token; i logits servono solo all'ultimo blocco. */
    for (int done = 0; done < n;) {
        int chunk = n - done < c->n_batch ? n - done : c->n_batch;
        int last = done + chunk == n;
        forward(c, tokens + done, chunk, last ? (c->logits_all ? chunk : 1) : 0);
        done += chunk;
    }
    c->n_logits = c->logits_all ? n : 1;
    return 0;
}

const float *engine_logits(const engine_ctx *c) { return c && c->n_logits ? c->logits : NULL; }
int32_t engine_n_logits(const engine_ctx *c) { return c ? c->n_logits : 0; }
int32_t engine_n_past(const engine_ctx *c) { return c ? c->n_past : 0; }
const char *engine_last_error(const engine_ctx *c) { return c ? c->err : "contesto NULL"; }

void engine_reset(engine_ctx *c) {
    if (!c) return;
    c->n_past = 0;
    c->n_logits = 0;
}

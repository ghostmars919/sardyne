#define _POSIX_C_SOURCE 200809L

#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_MAX_ALIGNMENT (1u << 20)

struct gguf_file {
    const uint8_t *map;
    size_t size;
    uint32_t version;
    uint32_t alignment;
    uint64_t data_offset;
    uint64_t n_kv;
    uint64_t n_tensors;
    gguf_kv *kv;
    gguf_tensor *tensors;
};

/* ---------------------------------------------------------------- tipi ggml */

static const struct {
    const char *name;
    uint32_t block; /* elementi per blocco */
    uint32_t size;  /* byte per blocco */
} GGML_TYPES[] = {
    [0] = {"F32", 1, 4},       [1] = {"F16", 1, 2},       [2] = {"Q4_0", 32, 18},
    [3] = {"Q4_1", 32, 20},    [6] = {"Q5_0", 32, 22},    [7] = {"Q5_1", 32, 24},
    [8] = {"Q8_0", 32, 34},    [9] = {"Q8_1", 32, 36},    [10] = {"Q2_K", 256, 84},
    [11] = {"Q3_K", 256, 110}, [12] = {"Q4_K", 256, 144}, [13] = {"Q5_K", 256, 176},
    [14] = {"Q6_K", 256, 210}, [15] = {"Q8_K", 256, 292}, [16] = {"IQ2_XXS", 256, 66},
    [17] = {"IQ2_XS", 256, 74}, [18] = {"IQ3_XXS", 256, 98}, [19] = {"IQ1_S", 256, 50},
    [20] = {"IQ4_NL", 32, 18}, [21] = {"IQ3_S", 256, 110}, [22] = {"IQ2_S", 256, 82},
    [23] = {"IQ4_XS", 256, 136}, [24] = {"I8", 1, 1},     [25] = {"I16", 1, 2},
    [26] = {"I32", 1, 4},      [27] = {"I64", 1, 8},      [28] = {"F64", 1, 8},
    [29] = {"IQ1_M", 256, 56}, [30] = {"BF16", 1, 2},     [34] = {"TQ1_0", 256, 54},
    [35] = {"TQ2_0", 256, 66}, [39] = {"MXFP4", 32, 17},
};
#define N_GGML_TYPES (sizeof(GGML_TYPES) / sizeof(GGML_TYPES[0]))

const char *gguf_type_name(uint32_t t) {
    return t < N_GGML_TYPES && GGML_TYPES[t].name ? GGML_TYPES[t].name : NULL;
}

/* Dimensione in byte dei valori scalari dei metadati (0 = non scalare). */
static size_t value_size(uint32_t vtype) {
    switch (vtype) {
    case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
    default: return 0;
    }
}

/* a * b senza overflow; 0 se ok. (Equivalente portabile di __builtin_mul_overflow.) */
static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b != 0 && a > UINT64_MAX / b) return -1;
    *out = a * b;
    return 0;
}

/* Chiavi e nomi dei tensori: UTF-8 valido e senza byte NUL. Le stesse regole del parser Python
 * (decode("utf-8") stretto): niente forme sovralunghe, surrogati o code point oltre U+10FFFF.
 * Così i due parser non possono interpretare lo stesso file in modo diverso. */
static int valid_name(const uint8_t *s, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        uint8_t c = s[i];
        if (c < 0x80) {
            if (c == 0) return 0;
            i++;
            continue;
        }
        uint32_t cp, len;
        if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        else return 0;
        if (n - i < len) return 0;
        for (uint32_t k = 1; k < len; k++) {
            uint8_t cc = s[i + k];
            if ((cc & 0xC0) != 0x80) return 0;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000)) return 0;
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
        i += len;
    }
    return 1;
}

static int host_is_little_endian(void) {
    const uint16_t x = 1;
    uint8_t b;
    memcpy(&b, &x, 1);
    return b == 1;
}

/* ---------------------------------------------------------------- lettura con controllo dei limiti */

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
    char *err;
    size_t err_len;
} reader;

static int fail(reader *r, const char *fmt, ...) {
    if (r->err && r->err_len) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(r->err, r->err_len, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static int need(reader *r, uint64_t n) {
    if (n > r->size - r->pos) return fail(r, "file troncato all'offset %zu (servono %llu byte)",
                                          r->pos, (unsigned long long)n);
    return 0;
}

/* Il file può avere qualsiasi allineamento: memcpy evita letture non allineate.
 * Il formato è little-endian e l'host deve esserlo (controllato in gguf_open). */
static int read_bytes(reader *r, void *out, size_t n) {
    if (need(r, n)) return -1;
    memcpy(out, r->base + r->pos, n);
    r->pos += n;
    return 0;
}

static int read_u32(reader *r, uint32_t *v) { return read_bytes(r, v, 4); }
static int read_u64(reader *r, uint64_t *v) { return read_bytes(r, v, 8); }

/* Legge una chiave o un nome di tensore e ne restituisce una copia terminata da NUL. */
static char *read_name_dup(reader *r, const char *what) {
    uint64_t n;
    if (read_u64(r, &n) || need(r, n)) return NULL;
    if (!valid_name(r->base + r->pos, n)) {
        fail(r, "%s all'offset %zu: UTF-8 non valido o byte NUL", what, r->pos);
        return NULL;
    }
    char *s = malloc(n + 1);
    if (!s) { fail(r, "memoria esaurita"); return NULL; }
    memcpy(s, r->base + r->pos, n);
    s[n] = '\0';
    r->pos += n;
    return s;
}

static int read_value(reader *r, gguf_kv *kv) {
    if (kv->type == GGUF_TYPE_STRING) {
        if (read_u64(r, &kv->n) || need(r, kv->n)) return -1;
        kv->ptr = r->base + r->pos;
        r->pos += kv->n;
        return 0;
    }
    if (kv->type == GGUF_TYPE_ARRAY) {
        if (read_u32(r, &kv->arr_type) || read_u64(r, &kv->n)) return -1;
        kv->ptr = r->base + r->pos;
        size_t es = value_size(kv->arr_type);
        if (es) {
            if (kv->n > (r->size - r->pos) / es) return fail(r, "array oltre la fine del file");
            r->pos += kv->n * es;
            return 0;
        }
        if (kv->arr_type != GGUF_TYPE_STRING) return fail(r, "array di tipo %u non supportato", kv->arr_type);
        /* Ogni stringa occupa almeno 8 byte: limita n prima di allocare. */
        if (kv->n > (r->size - r->pos) / 8) return fail(r, "array di stringhe oltre la fine del file");
        kv->str_pos = malloc((kv->n ? kv->n : 1) * sizeof(uint64_t));
        if (!kv->str_pos) return fail(r, "memoria esaurita");
        for (uint64_t j = 0; j < kv->n; j++) {
            uint64_t len;
            kv->str_pos[j] = r->pos;
            if (read_u64(r, &len) || need(r, len)) return -1;
            r->pos += len;
        }
        return 0;
    }
    size_t es = value_size(kv->type);
    if (!es) return fail(r, "tipo di valore sconosciuto %u", kv->type);
    if (need(r, es)) return -1;
    kv->ptr = r->base + r->pos;
    kv->n = 1;
    r->pos += es;
    return 0;
}

/* ---------------------------------------------------------------- duplicati */

static int cmp_names(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Ordina una copia dei puntatori ai nomi e confronta i vicini: O(n log n) anche con file ostili. */
static int find_duplicate(reader *r, const char **names, uint64_t n, const char *what) {
    if (n < 2) return 0;
    qsort(names, n, sizeof(*names), cmp_names);
    for (uint64_t i = 1; i < n; i++)
        if (strcmp(names[i - 1], names[i]) == 0) return fail(r, "%s duplicato: %s", what, names[i]);
    return 0;
}

static int check_duplicates(gguf_file *f, reader *r) {
    uint64_t n = f->n_kv > f->n_tensors ? f->n_kv : f->n_tensors;
    const char **names = malloc((n ? n : 1) * sizeof(*names));
    if (!names) return fail(r, "memoria esaurita");
    for (uint64_t i = 0; i < f->n_kv; i++) names[i] = f->kv[i].key;
    int rc = find_duplicate(r, names, f->n_kv, "metadato");
    if (rc == 0) {
        for (uint64_t i = 0; i < f->n_tensors; i++) names[i] = f->tensors[i].name;
        rc = find_duplicate(r, names, f->n_tensors, "tensore");
    }
    free(names);
    return rc;
}

/* ---------------------------------------------------------------- apertura */

void gguf_close(gguf_file *f) {
    if (!f) return;
    if (f->kv) {
        for (uint64_t i = 0; i < f->n_kv; i++) {
            free((char *)f->kv[i].key);
            free(f->kv[i].str_pos);
        }
        free(f->kv);
    }
    if (f->tensors) {
        for (uint64_t i = 0; i < f->n_tensors; i++) free((char *)f->tensors[i].name);
        free(f->tensors);
    }
    if (f->map) munmap((void *)f->map, f->size);
    free(f);
}

static int parse(gguf_file *f, reader *r) {
    char magic[4];
    if (read_bytes(r, magic, 4)) return -1;
    if (memcmp(magic, "GGUF", 4) != 0) return fail(r, "non è un file GGUF (magic errato)");
    if (read_u32(r, &f->version)) return -1;
    if ((f->version & 0xFFFF) == 0 && (f->version >> 24) >= 1 && (f->version >> 24) <= 3)
        return fail(r, "file GGUF big-endian non supportato");
    if (f->version != 2 && f->version != 3) return fail(r, "versione GGUF %u non supportata", f->version);

    uint64_t n_tensors, n_kv;
    if (read_u64(r, &n_tensors) || read_u64(r, &n_kv)) return -1;
    /* Controlli di plausibilità prima di allocare: ogni voce occupa almeno qualche byte. */
    if (n_kv > r->size / 12 || n_tensors > r->size / 32) return fail(r, "contatori implausibili");

    f->kv = calloc(n_kv ? n_kv : 1, sizeof(gguf_kv));
    f->tensors = calloc(n_tensors ? n_tensors : 1, sizeof(gguf_tensor));
    if (!f->kv || !f->tensors) return fail(r, "memoria esaurita");

    f->alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < n_kv; i++) {
        gguf_kv *kv = &f->kv[i];
        if (!(kv->key = read_name_dup(r, "chiave"))) return -1;
        f->n_kv = i + 1; /* da qui in poi gguf_close libera anche questa voce */
        if (read_u32(r, &kv->type) || read_value(r, kv)) return -1;
        if (strcmp(kv->key, "general.alignment") == 0) {
            uint64_t a;
            if (gguf_kv_get_u64(f, (int64_t)i, &a) || a == 0 || a > GGUF_MAX_ALIGNMENT || (a & (a - 1)))
                return fail(r, "general.alignment non valido");
            f->alignment = (uint32_t)a;
        }
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor *t = &f->tensors[i];
        if (!(t->name = read_name_dup(r, "nome di tensore"))) return -1;
        f->n_tensors = i + 1;
        if (read_u32(r, &t->n_dims)) return -1;
        if (t->n_dims < 1 || t->n_dims > GGUF_MAX_DIMS) return fail(r, "%s: n_dims=%u", t->name, t->n_dims);
        for (uint32_t d = 0; d < GGUF_MAX_DIMS; d++) {
            t->ne[d] = 1;
            if (d < t->n_dims && read_u64(r, &t->ne[d])) return -1;
            if (t->ne[d] == 0) return fail(r, "%s: dimensione %u nulla", t->name, d);
        }
        if (read_u32(r, &t->type) || read_u64(r, &t->offset)) return -1;
        if (!gguf_type_name(t->type)) return fail(r, "%s: tipo ggml sconosciuto %u", t->name, t->type);

        uint32_t block = GGML_TYPES[t->type].block;
        if (t->ne[0] % block) return fail(r, "%s: ne[0] non multiplo del blocco %u", t->name, block);
        uint64_t n = 1;
        for (uint32_t d = 0; d < GGUF_MAX_DIMS; d++)
            if (mul_u64(n, t->ne[d], &n)) return fail(r, "%s: dimensioni troppo grandi", t->name);
        if (mul_u64(n / block, GGML_TYPES[t->type].size, &t->nbytes))
            return fail(r, "%s: dimensioni troppo grandi", t->name);
    }

    if (check_duplicates(f, r)) return -1;

    /* La zona dati inizia al primo multiplo dell'allineamento dopo l'header. */
    f->data_offset = (r->pos + f->alignment - 1) / f->alignment * f->alignment;
    if (f->data_offset > r->size) return fail(r, "zona dati oltre la fine del file");
    uint64_t data_size = r->size - f->data_offset;

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        gguf_tensor *t = &f->tensors[i];
        if (t->offset % f->alignment) return fail(r, "%s: offset non allineato", t->name);
        if (t->offset > data_size || t->nbytes > data_size - t->offset)
            return fail(r, "%s: dati oltre la fine del file", t->name);
        t->data = f->map + f->data_offset + t->offset;
    }
    return 0;
}

void gguf_prefetch(const void *p, size_t n) {
    /* Il kernel tronca ogni richiesta a circa la dimensione massima di un I/O: a blocchi da 1 MiB. */
    enum { CHUNK = 1 << 20 };
    uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE), b = (uintptr_t)p & ~(page - 1), end = (uintptr_t)p + n;
    for (; b < end; b += CHUNK)
        posix_madvise((void *)b, end - b < CHUNK ? end - b : CHUNK, POSIX_MADV_WILLNEED);
}

gguf_file *gguf_open(const char *path, char *err, size_t err_len) {
    reader r = {.err = err, .err_len = err_len};
    if (err && err_len) err[0] = '\0';
    if (!host_is_little_endian()) { fail(&r, "host big-endian non supportato"); return NULL; }

    int fd = open(path, O_RDONLY);
    if (fd < 0) { fail(&r, "impossibile aprire %s: %s", path, strerror(errno)); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fail(&r, "impossibile leggere la dimensione di %s: %s", path, strerror(errno));
        close(fd);
        return NULL;
    }
    if (st.st_size <= 0 || (uintmax_t)st.st_size > SIZE_MAX) {
        close(fd);
        fail(&r, "%s: dimensione non valida o troppo grande per questa piattaforma", path);
        return NULL;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    int map_errno = errno;
    close(fd); /* la mappatura resta valida anche dopo la chiusura del descrittore */
    if (map == MAP_FAILED) { fail(&r, "mmap fallita su %s: %s", path, strerror(map_errno)); return NULL; }

    gguf_file *f = calloc(1, sizeof(*f));
    if (!f) { munmap(map, (size_t)st.st_size); fail(&r, "memoria esaurita"); return NULL; }
    f->map = map;
    f->size = (size_t)st.st_size;
    r.base = f->map;
    r.size = f->size;
    if (parse(f, &r) != 0) {
        gguf_close(f);
        return NULL;
    }
    return f;
}

/* ---------------------------------------------------------------- accesso
 * Tutti gli accessori tollerano f == NULL (restituiscono 0, NULL o -1). */

uint32_t gguf_version(const gguf_file *f) { return f ? f->version : 0; }
uint32_t gguf_alignment(const gguf_file *f) { return f ? f->alignment : 0; }
uint64_t gguf_data_offset(const gguf_file *f) { return f ? f->data_offset : 0; }
uint64_t gguf_n_kv(const gguf_file *f) { return f ? f->n_kv : 0; }
uint64_t gguf_n_tensors(const gguf_file *f) { return f ? f->n_tensors : 0; }

const gguf_kv *gguf_get_kv(const gguf_file *f, uint64_t i) {
    return f && i < f->n_kv ? &f->kv[i] : NULL;
}

const gguf_tensor *gguf_get_tensor(const gguf_file *f, uint64_t i) {
    return f && i < f->n_tensors ? &f->tensors[i] : NULL;
}

int64_t gguf_find_kv(const gguf_file *f, const char *key) {
    if (!f || !key) return -1;
    for (uint64_t i = 0; i < f->n_kv; i++)
        if (strcmp(f->kv[i].key, key) == 0) return (int64_t)i;
    return -1;
}

int64_t gguf_find_tensor(const gguf_file *f, const char *name) {
    if (!f || !name) return -1;
    for (uint64_t i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0) return (int64_t)i;
    return -1;
}

static const gguf_kv *kv_at(const gguf_file *f, int64_t i) {
    return f && i >= 0 && (uint64_t)i < f->n_kv ? &f->kv[i] : NULL;
}

int gguf_kv_get_u64(const gguf_file *f, int64_t i, uint64_t *out) {
    const gguf_kv *kv = kv_at(f, i);
    if (!kv || !out) return -1;
    union { uint8_t u8; int8_t i8; uint16_t u16; int16_t i16; uint32_t u32; int32_t i32;
            uint64_t u64; int64_t i64; } v;
    size_t es = value_size(kv->type);
    if (!es || kv->type == GGUF_TYPE_FLOAT32 || kv->type == GGUF_TYPE_FLOAT64 || kv->type == GGUF_TYPE_BOOL)
        return -1;
    memcpy(&v, kv->ptr, es);
    int64_t s;
    switch (kv->type) {
    case GGUF_TYPE_UINT8: *out = v.u8; return 0;
    case GGUF_TYPE_UINT16: *out = v.u16; return 0;
    case GGUF_TYPE_UINT32: *out = v.u32; return 0;
    case GGUF_TYPE_UINT64: *out = v.u64; return 0;
    case GGUF_TYPE_INT8: s = v.i8; break;
    case GGUF_TYPE_INT16: s = v.i16; break;
    case GGUF_TYPE_INT32: s = v.i32; break;
    default: s = v.i64; break;
    }
    if (s < 0) return -1;
    *out = (uint64_t)s;
    return 0;
}

int gguf_kv_get_f64(const gguf_file *f, int64_t i, double *out) {
    const gguf_kv *kv = kv_at(f, i);
    if (!kv || !out) return -1;
    if (kv->type == GGUF_TYPE_FLOAT32) {
        float v;
        memcpy(&v, kv->ptr, 4);
        *out = v;
        return 0;
    }
    if (kv->type == GGUF_TYPE_FLOAT64) {
        memcpy(out, kv->ptr, 8);
        return 0;
    }
    return -1;
}

const char *gguf_kv_get_str(const gguf_file *f, int64_t i, uint64_t *len) {
    const gguf_kv *kv = kv_at(f, i);
    if (!kv || !len || kv->type != GGUF_TYPE_STRING) return NULL;
    *len = kv->n;
    return (const char *)kv->ptr;
}

const char *gguf_kv_get_arr_str(const gguf_file *f, int64_t i, uint64_t j, uint64_t *len) {
    const gguf_kv *kv = kv_at(f, i);
    if (!kv || !len || kv->type != GGUF_TYPE_ARRAY || kv->arr_type != GGUF_TYPE_STRING || j >= kv->n)
        return NULL;
    memcpy(len, f->map + kv->str_pos[j], 8);
    return (const char *)(f->map + kv->str_pos[j] + 8);
}

#ifndef GGUF_H
#define GGUF_H

/* Parser GGUF in sola lettura con mmap.
 *
 * Il file viene mappato in memoria una volta: i dati dei tensori restano nel file e
 * `gguf_tensor.data` punta direttamente dentro la mappatura (zero copie). Il sistema
 * operativo carica le pagine quando servono e può scaricarle sotto pressione di memoria.
 * Il formato è descritto in engine/loader/gguf.py.
 *
 * Il file viene validato una sola volta, all'apertura: chiavi e nomi UTF-8 senza NUL e senza
 * duplicati, dimensioni non nulle e senza overflow, dati dentro il file e allineati.
 * Limite noto (come in llama.cpp): il file NON deve essere troncato o riscritto sul posto mentre
 * è aperto. Le pagine oltre la nuova fine non esistono più e leggerle termina il processo con
 * SIGBUS. Chi scrive file GGUF deve sostituirli in modo atomico (nuovo file + rename), come fa
 * engine/loader/gguf_writer.py: chi ha già mappato il vecchio file continua a vederlo intatto.
 */

#include <stddef.h>
#include <stdint.h>

enum gguf_value_type {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

#define GGUF_MAX_DIMS 4

typedef struct {
    const char *name;   /* copia terminata da NUL */
    uint32_t n_dims;
    uint32_t type;      /* tipo ggml (0 = F32, 1 = F16, 8 = Q8_0, 30 = BF16, ...) */
    uint64_t ne[GGUF_MAX_DIMS]; /* ne[0] = dimensione più interna; quelle non usate valgono 1 */
    uint64_t offset;    /* relativo all'inizio della zona dati */
    uint64_t nbytes;
    const void *data;   /* puntatore dentro la mmap */
} gguf_tensor;

typedef struct {
    const char *key;    /* copia terminata da NUL */
    uint32_t type;      /* enum gguf_value_type */
    uint32_t arr_type;  /* tipo degli elementi, solo per gli array */
    uint64_t n;         /* elementi (array) o byte (stringhe) */
    const uint8_t *ptr; /* valore dentro la mmap: scalare, byte della stringa o primo elemento */
    uint64_t *str_pos;  /* array di stringhe: posizione nel file di ogni elemento */
} gguf_kv;

typedef struct gguf_file gguf_file;

/* Restituisce NULL in caso di errore e scrive il motivo in err. */
gguf_file *gguf_open(const char *path, char *err, size_t err_len);
void gguf_close(gguf_file *f);
/* Chiede al sistema di caricare in anticipo n byte della mmap, senza aspettare la lettura. */
void gguf_prefetch(const void *p, size_t n);

uint32_t gguf_version(const gguf_file *f);
uint32_t gguf_alignment(const gguf_file *f);
uint64_t gguf_data_offset(const gguf_file *f);
uint64_t gguf_n_kv(const gguf_file *f);
uint64_t gguf_n_tensors(const gguf_file *f);

const gguf_kv *gguf_get_kv(const gguf_file *f, uint64_t i);
int64_t gguf_find_kv(const gguf_file *f, const char *key); /* -1 se assente */
const gguf_tensor *gguf_get_tensor(const gguf_file *f, uint64_t i);
int64_t gguf_find_tensor(const gguf_file *f, const char *name);

/* Getter tipizzati: 0 se ok, -1 se il tipo non è compatibile. */
int gguf_kv_get_u64(const gguf_file *f, int64_t i, uint64_t *out); /* qualsiasi intero >= 0 */
int gguf_kv_get_f64(const gguf_file *f, int64_t i, double *out);   /* FLOAT32 o FLOAT64 */
const char *gguf_kv_get_str(const gguf_file *f, int64_t i, uint64_t *len); /* NON terminata da NUL */
const char *gguf_kv_get_arr_str(const gguf_file *f, int64_t i, uint64_t j, uint64_t *len);

const char *gguf_type_name(uint32_t ggml_type);

#endif

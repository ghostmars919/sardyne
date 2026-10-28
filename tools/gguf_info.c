/* Riepilogo di un file GGUF usando solo il parser C (csrc/gguf.c).
 * Uso: build/gguf-info file.gguf
 */
#include <stdio.h>

#include "../csrc/gguf.h"

#define MAX_PRINT 200

/* Le stringhe vengono dal file: si stampano al massimo MAX_PRINT byte e i caratteri di controllo
 * diventano \xNN, così un file non può inviare sequenze di escape al terminale. */
static void print_safe(const char *s, uint64_t len) {
    uint64_t n = len > MAX_PRINT ? MAX_PRINT : len;
    for (uint64_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x20 || c == 0x7F) printf("\\x%02x", c);
        else putchar(c);
    }
    if (len > n) printf("... (%llu byte)", (unsigned long long)len);
}

static void print_kv(const gguf_file *f, const char *key) {
    int64_t i = gguf_find_kv(f, key);
    uint64_t u, len;
    double d;
    const char *s;
    if (i < 0) return;
    if (gguf_kv_get_u64(f, i, &u) == 0) printf("  %-40s %llu\n", key, (unsigned long long)u);
    else if (gguf_kv_get_f64(f, i, &d) == 0) printf("  %-40s %g\n", key, d);
    else if ((s = gguf_kv_get_str(f, i, &len))) {
        printf("  %-40s ", key);
        print_safe(s, len);
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "uso: %s file.gguf\n", argv[0]);
        return 2;
    }
    char err[256];
    gguf_file *f = gguf_open(argv[1], err, sizeof(err));
    if (!f) {
        fprintf(stderr, "errore: %s\n", err);
        return 1;
    }
    printf("GGUF v%u, %llu metadati, %llu tensori, dati da offset %llu\n", gguf_version(f),
           (unsigned long long)gguf_n_kv(f), (unsigned long long)gguf_n_tensors(f),
           (unsigned long long)gguf_data_offset(f));

    uint64_t len;
    const char *arch = gguf_kv_get_str(f, gguf_find_kv(f, "general.architecture"), &len);
    print_kv(f, "general.architecture");
    print_kv(f, "general.name");
    if (arch && len < 64) { /* le chiavi sono "<arch>.<suffisso>" */
        static const char *suffixes[] = {"block_count", "embedding_length", "feed_forward_length",
                                         "attention.head_count", "attention.head_count_kv",
                                         "rope.freq_base", "context_length"};
        for (size_t k = 0; k < sizeof(suffixes) / sizeof(suffixes[0]); k++) {
            char key[128];
            snprintf(key, sizeof(key), "%.*s.%s", (int)len, arch, suffixes[k]);
            print_kv(f, key);
        }
    }

    /* Byte e numero di tensori per tipo ggml. */
    uint64_t bytes[64] = {0}, count[64] = {0}, total = 0;
    for (uint64_t i = 0; i < gguf_n_tensors(f); i++) {
        const gguf_tensor *t = gguf_get_tensor(f, i);
        if (t->type < 64) {
            bytes[t->type] += t->nbytes;
            count[t->type]++;
        }
        total += t->nbytes;
    }
    printf("\n  tipo     tensori        MB\n");
    for (int t = 0; t < 64; t++)
        if (count[t]) printf("  %-8s %7llu %9.1f\n", gguf_type_name(t), (unsigned long long)count[t], bytes[t] / 1e6);
    printf("  totale   %7llu %9.1f\n", (unsigned long long)gguf_n_tensors(f), total / 1e6);

    int64_t ie = gguf_find_tensor(f, "token_embd.weight");
    const gguf_tensor *emb = ie >= 0 ? gguf_get_tensor(f, (uint64_t)ie) : NULL;
    if (emb)
        printf("\n  token_embd.weight: %s, ne = [%llu, %llu], dati a %p (dentro la mmap)\n",
               gguf_type_name(emb->type), (unsigned long long)emb->ne[0], (unsigned long long)emb->ne[1], emb->data);
    gguf_close(f);
    return 0;
}

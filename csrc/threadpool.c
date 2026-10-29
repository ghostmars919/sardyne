#define _POSIX_C_SOURCE 200809L

#include "threadpool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#define SPIN_LIMIT (1 << 15) /* iterazioni di attesa attiva prima di dormire (~1 ms) */
#define MAX_THREADS 256

struct threadpool {
    int n;
    pthread_t *threads;
    tp_fn fn;
    void *arg;
    atomic_uint gen;  /* incrementato a ogni operazione: i thread aspettano che cambi */
    atomic_int done;  /* thread che hanno finito l'operazione corrente */
    atomic_int sleeping;
    atomic_int stop;
    pthread_mutex_t mu;
    pthread_cond_t cv;
};

typedef struct {
    threadpool *p;
    int ith;
} worker_arg;

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

static void *worker(void *v) {
    worker_arg wa = *(worker_arg *)v;
    free(v);
    threadpool *p = wa.p;
    unsigned seen = 0;
    for (;;) {
        unsigned g;
        int spins = 0;
        while ((g = atomic_load(&p->gen)) == seen) {
            if (++spins < SPIN_LIMIT) {
                cpu_relax();
                continue;
            }
            /* Attesa lunga: si dorme. sleeping e gen usano l'ordinamento seq_cst (il default):
             * o tp_run vede sleeping > 0 e sveglia, o questo thread vede il nuovo gen. */
            pthread_mutex_lock(&p->mu);
            atomic_fetch_add(&p->sleeping, 1);
            while (atomic_load(&p->gen) == seen) pthread_cond_wait(&p->cv, &p->mu);
            atomic_fetch_sub(&p->sleeping, 1);
            pthread_mutex_unlock(&p->mu);
            spins = 0;
        }
        seen = g;
        if (atomic_load(&p->stop)) return NULL;
        p->fn(p->arg, wa.ith, p->n);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

int tp_default_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (n > MAX_THREADS ? MAX_THREADS : (int)n);
}

static void wake_all(threadpool *p) {
    if (atomic_load(&p->sleeping) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
}

threadpool *tp_new(int n_threads) {
    if (n_threads < 1 || n_threads > MAX_THREADS) return NULL;
    threadpool *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n = n_threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    if (n_threads > 1) {
        p->threads = calloc((size_t)n_threads - 1, sizeof(pthread_t));
        if (!p->threads) {
            tp_free(p);
            return NULL;
        }
        for (int i = 1; i < n_threads; i++) {
            worker_arg *wa = malloc(sizeof(*wa));
            if (wa) {
                wa->p = p; /* prima di pthread_create: il thread lo legge subito */
                wa->ith = i;
            }
            if (!wa || pthread_create(&p->threads[i - 1], NULL, worker, wa) != 0) {
                free(wa);
                p->n = i; /* tp_free ferma solo i thread già creati */
                tp_free(p);
                return NULL;
            }
        }
    }
    return p;
}

void tp_free(threadpool *p) {
    if (!p) return;
    if (p->threads) {
        atomic_store(&p->stop, 1);
        atomic_fetch_add(&p->gen, 1);
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
        for (int i = 1; i < p->n; i++) pthread_join(p->threads[i - 1], NULL);
        free(p->threads);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p);
}

int tp_n_threads(const threadpool *p) { return p ? p->n : 1; }

void tp_run(threadpool *p, tp_fn fn, void *arg) {
    if (!p || p->n == 1) {
        fn(arg, 0, 1);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_fetch_add(&p->gen, 1); /* pubblica fn/arg (seq_cst, quindi anche release) */
    wake_all(p);
    fn(arg, 0, p->n);
    while (atomic_load_explicit(&p->done, memory_order_acquire) < p->n - 1) cpu_relax();
}

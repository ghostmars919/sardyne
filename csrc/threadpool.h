#ifndef THREADPOOL_H
#define THREADPOOL_H

/* Pool di thread per parallelizzare i kernel di un passo del modello.
 *
 * tp_run(pool, fn, arg) esegue fn(arg, ith, nth) su tutti gli nth thread (il chiamante è ith = 0)
 * e ritorna quando tutti hanno finito: è una barriera dopo ogni operazione.
 *
 * Un passo di decode lancia centinaia di operazioni brevi (decine di microsecondi): svegliare i
 * thread con una condition variable costerebbe quanto il lavoro stesso. Per questo i thread,
 * finita un'operazione, restano in attesa attiva per un po' (circa un millisecondo) e solo dopo
 * si addormentano.
 */

typedef struct threadpool threadpool;
typedef void (*tp_fn)(void *arg, int ith, int nth);

/* n_threads totali, compreso il chiamante; NULL in caso di errore. */
threadpool *tp_new(int n_threads);
void tp_free(threadpool *p);
int tp_n_threads(const threadpool *p);
void tp_run(threadpool *p, tp_fn fn, void *arg);

/* Numero di core disponibili (almeno 1). */
int tp_default_threads(void);

#endif

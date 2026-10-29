#ifndef OPS_INTERNAL_H
#define OPS_INTERNAL_H

/* Kernel specializzati per CPU, scelti a runtime da ops.c. Non fanno parte dell'API. */

#include <stdint.h>

typedef void (*op_matmul_rows_fn)(float *out, const float *x, const void *w, int n_tok, int n_in, int n_out,
                                  int row0, int row1);
/* Come sopra, con le attivazioni già quantizzate (xq). */
typedef void (*quant_matmul_rows_fn)(float *out, const void *xq, const void *w, int n_tok, int n_in, int n_out,
                                     int row0, int row1);

#if defined(__x86_64__) || defined(__i386__)
#define OPS_HAVE_AVX2 1
int op_cpu_has_avx2(void); /* AVX2 + FMA + F16C */
void op_matmul_rows_avx2_f32(float *, const float *, const void *, int, int, int, int, int);
void op_matmul_rows_avx2_f16(float *, const float *, const void *, int, int, int, int, int);
void op_matmul_rows_avx2_bf16(float *, const float *, const void *, int, int, int, int, int);
float op_dot_f32_avx2(const float *a, const float *b, int n);
void op_axpy_avx2(float *y, float a, const float *x, int n);
float quant_dot_q8_0_avx2(const void *w, const void *x, int n);
float quant_dot_q4_0_avx2(const void *w, const void *x, int n);
float quant_dot_q5_0_avx2(const void *w, const void *x, int n);
float quant_dot_q4_K_avx2(const void *w, const void *x, int n);
float quant_dot_q6_K_avx2(const void *w, const void *x, int n);
void quant_matmul_rows_q8_0_avx2(float *, const void *, const void *, int, int, int, int, int);
void quant_matmul_rows_q4_0_avx2(float *, const void *, const void *, int, int, int, int, int);
void quant_matmul_rows_q5_0_avx2(float *, const void *, const void *, int, int, int, int, int);
void quant_matmul_rows_q4_K_avx2(float *, const void *, const void *, int, int, int, int, int);
void quant_matmul_rows_q6_K_avx2(float *, const void *, const void *, int, int, int, int, int);
#endif

#endif

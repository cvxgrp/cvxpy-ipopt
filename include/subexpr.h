/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SUBEXPR_H
#define SUBEXPR_H

#include "expr.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"

/* Forward declaration */
struct int_double_pair;

/* Type-specific expression structures that "inherit" from expr */

/* Linear operator: y = A * x + b */
typedef struct linear_op_expr
{
    expr base;
    CSC_Matrix *A_csc;
    CSR_Matrix *A_csr;
    double *b; /* constant offset vector (NULL if no offset) */
} linear_op_expr;

/* Power: y = x^p */
typedef struct power_expr
{
    expr base;
    double p;
} power_expr;

/* Quadratic form: y = x'*Q*x */
typedef struct quad_form_expr
{
    expr base;
    CSR_Matrix *Q;
} quad_form_expr;

/* Sum reduction along an axis */
typedef struct sum_expr
{
    expr base;
    int axis;
    int *idx_map; /* maps child nnz to summed-row positions */
} sum_expr;

/* trace */
typedef struct trace_expr
{
    expr base;
    int *idx_map; /* maps child nnz to summed-row positions */
} trace_expr;

/* Product of all entries */
typedef struct prod_expr
{
    expr base;
    int num_of_zeros;
    int zero_index;      /* index of zero element when num_of_zeros == 1 */
    double prod_nonzero; /* product of non-zero elements */
} prod_expr;

/* Product of entries along axis=0 (columnwise products) or axis = 1 (rowwise
 * products) */
typedef struct prod_axis
{
    expr base;
    int *num_of_zeros; /* num of zeros for each column / row depending on the axis*/
    int *zero_index;   /* stores idx of zero element per column / row */
    double *prod_nonzero; /* product of non-zero elements per column / row */
} prod_axis;

/* Horizontal stack (concatenate) */
typedef struct hstack_expr
{
    expr base;
    expr **args;
    int n_args;
    CSR_Matrix *CSR_work; /* for summing Hessians of children */
} hstack_expr;

/* Elementwise multiplication */
typedef struct elementwise_mult_expr
{
    expr base;
    CSR_Matrix *CSR_work1;
    CSR_Matrix *CSR_work2;
} elementwise_mult_expr;

/* Left matrix multiplication: y = A * f(x) where f(x) is an expression. Note that
here A does not have global column indices but it is a local matrix. This is an
important distinction compared to linear_op_expr. */
typedef struct left_matmul_expr
{
    expr base;
    CSR_Matrix *A;
    CSR_Matrix *AT;
    CSC_Matrix *CSC_work;
} left_matmul_expr;

/* Right matrix multiplication: y = f(x) * A where f(x) is an expression.
 * f(x) has shape p x n, A has shape n x q, output y has shape p x q.
 * Uses vec(y) = B * vec(f(x)) where B = A^T kron I_p. */
typedef struct right_matmul_expr
{
    expr base;
    CSR_Matrix *B;  /* B = A^T kron I_p */
    CSR_Matrix *BT; /* B^T for backpropagating Hessian weights */
    CSC_Matrix *CSC_work;
} right_matmul_expr;

/* Constant scalar multiplication: y = a * child where a is a constant double */
typedef struct const_scalar_mult_expr
{
    expr base;
    double a;
} const_scalar_mult_expr;

/* Constant vector elementwise multiplication: y = a \circ child for constant a */
typedef struct const_vector_mult_expr
{
    expr base;
    double *a; /* length equals node->size */
} const_vector_mult_expr;

/* Index/slicing: y = child[indices] where indices is a list of flat positions */
typedef struct index_expr
{
    expr base;
    int *indices;        /* Flattened indices to select (owned, copied) */
    int n_idxs;          /* Number of selected elements */
    bool has_duplicates; /* True if indices have duplicates (affects Hessian path) */
} index_expr;

/* Broadcast types */
typedef enum
{
    BROADCAST_ROW,   /* (1, n) -> (m, n) */
    BROADCAST_COL,   /* (m, 1) -> (m, n) */
    BROADCAST_SCALAR /* (1, 1) -> (m, n) */
} broadcast_type;

typedef struct broadcast_expr
{
    expr base;
    broadcast_type type;
} broadcast_expr;

#endif /* SUBEXPR_H */

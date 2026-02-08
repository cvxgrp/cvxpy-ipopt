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
#include "affine.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reshape changes the shape of an expression without permuting data.
 * Only Fortran (column-major) order is supported, where reshape is a no-op
 * for the underlying data layout. */

static void forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    memcpy(node->value, node->left->value, node->size * sizeof(double));
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    x->jacobian_init(x);
    node->jacobian = new_csr_matrix(node->size, node->n_vars, x->jacobian->nnz);
    CSR_Matrix *jac = node->jacobian;
    memcpy(jac->p, x->jacobian->p, (x->size + 1) * sizeof(int));
    memcpy(jac->i, x->jacobian->i, x->jacobian->nnz * sizeof(int));
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    x->eval_jacobian(x);
    memcpy(node->jacobian->x, x->jacobian->x, x->jacobian->nnz * sizeof(double));
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    x->wsum_hess_init(x);
    node->wsum_hess =
        new_csr_matrix(x->wsum_hess->m, x->wsum_hess->n, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (x->wsum_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    x->eval_wsum_hess(x, w);
    memcpy(node->wsum_hess->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_reshape(expr *child, int d1, int d2)
{
    assert(d1 * d2 == child->size);
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}

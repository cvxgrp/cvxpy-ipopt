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
// SPDX-License-Identifier: Apache-2.0

#include "affine.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* diag_vec: converts a vector of size n into an n×n diagonal matrix.
 * In Fortran (column-major) order, element i of the input maps to
 * position i*(n+1) in the flattened output (the diagonal positions). */

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    int n = x->size;

    /* child's forward pass */
    x->forward(x, u);

    /* zero-initialize output */
    memset(node->value, 0, node->size * sizeof(double));

    /* place input elements on the diagonal */
    for (int i = 0; i < n; i++)
    {
        node->value[i * (n + 1)] = x->value[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    int n = x->size;
    x->jacobian_init(x);

    CSR_Matrix *Jx = x->jacobian;
    CSR_Matrix *J = new_csr_matrix(node->size, node->n_vars, Jx->nnz);

    /* Output has n^2 rows but only n diagonal positions are non-empty.
     * Diagonal position i is at row i*(n+1) in Fortran order. */
    int nnz = 0;
    int next_diag = 0;
    for (int row = 0; row < node->size; row++)
    {
        J->p[row] = nnz;
        if (row == next_diag)
        {
            int child_row = row / (n + 1);
            int len = Jx->p[child_row + 1] - Jx->p[child_row];
            memcpy(J->i + nnz, Jx->i + Jx->p[child_row], len * sizeof(int));
            nnz += len;
            next_diag += n + 1;
        }
    }
    J->p[node->size] = nnz;

    node->jacobian = J;
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    int n = x->size;
    x->eval_jacobian(x);

    CSR_Matrix *J = node->jacobian;
    CSR_Matrix *Jx = x->jacobian;

    /* Copy values from child row i to output diagonal row i*(n+1) */
    for (int i = 0; i < n; i++)
    {
        int out_row = i * (n + 1);
        int len = J->p[out_row + 1] - J->p[out_row];
        memcpy(J->x + J->p[out_row], Jx->x + Jx->p[i], len * sizeof(double));
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    x->wsum_hess_init(x);

    /* workspace for extracting diagonal weights */
    node->dwork = (double *) calloc(x->size, sizeof(double));

    /* Copy child's Hessian structure (diag_vec is linear, so its own Hessian is
     * zero) */
    CSR_Matrix *Hx = x->wsum_hess;
    node->wsum_hess = new_csr_matrix(Hx->m, Hx->n, Hx->nnz);
    memcpy(node->wsum_hess->p, Hx->p, (Hx->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, Hx->i, Hx->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    int n = x->size;

    /* Extract weights from diagonal positions of w (which has n^2 elements) */
    for (int i = 0; i < n; i++)
    {
        node->dwork[i] = w[i * (n + 1)];
    }

    /* Evaluate child's Hessian with extracted weights */
    x->eval_wsum_hess(x, node->dwork);
    memcpy(node->wsum_hess->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_diag_vec(expr *child)
{
    /* child must be a vector: either column (n, 1) or row (1, n) */
    assert(child->d1 == 1 || child->d2 == 1);

    /* n is the number of elements (works for both row and column vectors) */
    int n = child->size;
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, n, n, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}

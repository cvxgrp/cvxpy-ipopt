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
#include "bivariate.h"
#include "subexpr.h"
#include "utils/mini_numpy.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Implementation of matrix multiplication: Z = X @ Y
// where X is m x k and Y is k x n, producing Z which is m x n
// All matrices are stored in column-major order
// ------------------------------------------------------------------------------

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    mat_mat_mult(x->value, y->value, node->value, x->d1, x->d2, y->d2);
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* dimensions: X is m x k, Y is k x n, Z is m x n */
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    int nnz = m * n * 2 * k;
    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz);

    /* fill sparsity pattern */
    int nnz_idx = 0;
    for (int i = 0; i < node->size; i++)
    {
        /* Convert flat index to (row, col) in Z */
        int row = i % m;
        int col = i / m;

        node->jacobian->p[i] = nnz_idx;

        /* X has lower var_id */
        if (x->var_id < y->var_id)
        {
            /* sparsity pattern of kron(YT, I) for this row */
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = x->var_id + row + j * m;
            }

            /* sparsity pattern of kron(I, X) for this row */
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = y->var_id + col * k + j;
            }
        }
        else /* Y has lower var_id */
        {
            /* sparsity pattern of kron(I, X) for this row */
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = y->var_id + col * k + j;
            }

            /* sparsity pattern of kron(YT, I) for this row */
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = x->var_id + row + j * m;
            }
        }
    }
    node->jacobian->p[node->size] = nnz_idx;
    assert(nnz_idx == nnz);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* dimensions: X is m x k, Y is k x n, Z is m x n */
    int m = x->d1;
    int k = x->d2;
    double *Jx = node->jacobian->x;

    /* fill values row-by-row */
    for (int i = 0; i < node->size; i++)
    {
        int row = i % m; /* row in Z */
        int col = i / m; /* col in Z */
        int pos = node->jacobian->p[i];

        if (x->var_id < y->var_id)
        {
            /* contribution to this row from YT */
            memcpy(Jx + pos, y->value + col * k, k * sizeof(double));

            /* contribution to this row from X */
            for (int j = 0; j < k; j++)
            {
                Jx[pos + k + j] = x->value[row + j * m];
            }
        }
        else
        {
            /* contribution to this row from X */
            for (int j = 0; j < k; j++)
            {
                Jx[pos + j] = x->value[row + j * m];
            }

            /* contribution to this row from YT */
            memcpy(Jx + pos + k, y->value + col * k, k * sizeof(double));
        }
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* dimensions: X is m x k, Y is k x n, Z is m x n */
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    int total_nnz = 2 * m * k * n;
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, total_nnz);
    int nnz = 0;
    int *Hi = node->wsum_hess->i;
    int *Hp = node->wsum_hess->p;
    int start, i;

    if (x->var_id < y->var_id)
    {
        /* fill rows corresponding to x */
        for (i = 0; i < x->size; i++)
        {
            Hp[x->var_id + i] = nnz;
            start = y->var_id + (i / m);
            for (int col = 0; col < n; col++)
            {
                Hi[nnz++] = start + col * k;
            }
        }

        /* fill rows between x and y */
        for (i = x->var_id + x->size; i < y->var_id; i++)
        {
            Hp[i] = nnz;
        }

        /* fill rows corresponding to y */
        for (i = 0; i < y->size; i++)
        {
            Hp[y->var_id + i] = nnz;
            start = x->var_id + (i % k) * m;
            for (int row = 0; row < m; row++)
            {
                Hi[nnz++] = start + row;
            }
        }

        /* fill rows after y */
        for (i = y->var_id + y->size; i <= node->n_vars; i++)
        {
            Hp[i] = nnz;
        }
    }
    else
    {
        /* Y has lower var_id than X */
        /* fill rows corresponding to y */
        for (i = 0; i < y->size; i++)
        {
            Hp[y->var_id + i] = nnz;
            start = x->var_id + (i % k) * m;
            for (int row = 0; row < m; row++)
            {
                Hi[nnz++] = start + row;
            }
        }

        /* fill rows between y and x */
        for (i = y->var_id + y->size; i < x->var_id; i++)
        {
            Hp[i] = nnz;
        }

        /* fill rows corresponding to x */
        for (i = 0; i < x->size; i++)
        {
            Hp[x->var_id + i] = nnz;
            start = y->var_id + (i / m);
            for (int col = 0; col < n; col++)
            {
                Hi[nnz++] = start + col * k;
            }
        }

        /* fill rows after x */
        for (i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            Hp[i] = nnz;
        }
    }

    Hp[node->n_vars] = nnz;
    assert(nnz == total_nnz);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    expr *y = node->right;

    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    int offset = 0;

    double *Hx = node->wsum_hess->x;
    const double *w_temp;

    if (x->var_id < y->var_id)
    {
        /* rows corresponding to x */
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    Hx[offset++] = w[row + col * m];
                }
            }
        }

        /* rows corresponding to y */
        for (int col = 0; col < n; col++)
        {
            w_temp = w + col * m;
            for (int k_idx = 0; k_idx < k; k_idx++)
            {
                memcpy(Hx + offset, w_temp, m * sizeof(double));
                offset += m;
            }
        }
    }
    else
    {
        /* rows corresponding to y */
        for (int col = 0; col < n; col++)
        {
            w_temp = w + col * m;
            for (int k_idx = 0; k_idx < k; k_idx++)
            {
                memcpy(Hx + offset, w_temp, m * sizeof(double));
                offset += m;
            }
        }

        /* rows corresponding to x */
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    Hx[offset++] = w[row + col * m];
                }
            }
        }
    }
}

expr *new_matmul(expr *x, expr *y)
{
    /* Verify dimensions: x->d2 must equal y->d1 */
    if (x->d2 != y->d1)
    {
        fprintf(stderr,
                "Error in new_matmul: dimension mismatch. "
                "X is %d x %d, Y is %d x %d. X.d2 (%d) must equal Y.d1 (%d)\n",
                x->d1, x->d2, y->d1, y->d2, x->d2, y->d1);
        exit(1);
    }

    /* verify both are variables and not the same variable */
    if (x->var_id == NOT_A_VARIABLE || y->var_id == NOT_A_VARIABLE ||
        x->var_id == y->var_id)
    {
        fprintf(stderr, "Error in new_matmul: operands must be variables and not "
                        "the same variable\n");
        exit(1);
    }

    /* Allocate the expression node */
    expr *node = (expr *) calloc(1, sizeof(expr));

    /* Initialize with d1 = x->d1, d2 = y->d2 (result is m x n) */
    init_expr(node, x->d1, y->d2, x->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);

    /* Set children */
    node->left = x;
    node->right = y;
    expr_retain(x);
    expr_retain(y);

    return node;
}

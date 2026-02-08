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
#include "subexpr.h"
#include "utils/mini_numpy.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Broadcast expands an array to a larger shape by replicating along dimensions.
 * Supports three types:
 * 1. "row": (1, n) -> (m, n) - replicate rows
 * 2. "col": (m, 1) -> (m, n) - replicate columns
 * 3. "scalar": (1, 1) -> (m, n) - replicate in both dimensions
 */

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    broadcast_expr *bcast = (broadcast_expr *) node;

    x->forward(x, u);

    if (bcast->type == BROADCAST_ROW)
    {
        /* (1, n) -> (m, n): replicate row m times */
        for (int j = 0; j < node->d2; j++)
        {
            for (int i = 0; i < node->d1; i++)
            {
                node->value[i + j * node->d1] = x->value[j];
            }
        }
    }
    else if (bcast->type == BROADCAST_COL)
    {
        /* (m, 1) -> (m, n): replicate column n times */
        for (int j = 0; j < node->d2; j++)
        {
            memcpy(node->value + j * node->d1, x->value, node->d1 * sizeof(double));
        }
    }
    else
    {
        /* (1, 1) -> (m, n): fill with scalar value */
        for (int k = 0; k < node->size; k++)
        {
            node->value[k] = x->value[0];
        }
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    x->jacobian_init(x);
    broadcast_expr *bcast = (broadcast_expr *) node;
    int total_nnz;

    // --------------------------------------------------------------------
    //                     count number of nonzeros
    // --------------------------------------------------------------------
    if (bcast->type == BROADCAST_ROW)
    {
        /* Row broadcast: (1, n) -> (m, n) */
        total_nnz = x->jacobian->nnz * node->d1;
    }
    else if (bcast->type == BROADCAST_COL)
    {
        /* Column broadcast: (m, 1) -> (m, n) */
        total_nnz = x->jacobian->nnz * node->d2;
    }
    else
    {
        /* Scalar broadcast: (1, 1) -> (m, n) */
        total_nnz = x->jacobian->nnz * node->d1 * node->d2;
    }

    node->jacobian = new_csr_matrix(node->size, node->n_vars, total_nnz);

    // ---------------------------------------------------------------------
    //                 fill sparsity pattern
    // ---------------------------------------------------------------------
    CSR_Matrix *Jx = x->jacobian;
    CSR_Matrix *J = node->jacobian;
    J->nnz = 0;

    if (bcast->type == BROADCAST_ROW)
    {
        for (int i = 0; i < node->d2; i++)
        {
            int nnz_in_row = Jx->p[i + 1] - Jx->p[i];

            /* copy columns indices */
            tile_int(J->i + J->nnz, Jx->i + Jx->p[i], nnz_in_row, node->d1);

            /* set row pointers */
            for (int rep = 0; rep < node->d1; rep++)
            {
                J->p[i * node->d1 + rep] = J->nnz;
                J->nnz += nnz_in_row;
            }
        }
        J->p[node->size] = total_nnz;
    }
    else if (bcast->type == BROADCAST_COL)
    {

        /* copy column indices */
        tile_int(J->i, Jx->i, Jx->nnz, node->d2);

        /* set row pointers */
        int offset = 0;
        for (int i = 0; i < node->d2; i++)
        {
            for (int j = 0; j < node->d1; j++)
            {
                J->p[i * node->d1 + j] = offset;
                offset += Jx->p[1] - Jx->p[0];
            }
        }
        assert(offset == total_nnz);
        J->p[node->size] = total_nnz;
    }
    else
    {
        /* copy column indices */
        tile_int(J->i, Jx->i, Jx->nnz, node->d1 * node->d2);

        /* set row pointers */
        int offset = 0;
        int nnz = Jx->p[1] - Jx->p[0];
        for (int i = 0; i < node->d1 * node->d2; i++)
        {
            J->p[i] = offset;
            offset += nnz;
        }
        assert(offset == total_nnz);
        J->p[node->size] = total_nnz;
    }
}

static void eval_jacobian(expr *node)
{
    node->left->eval_jacobian(node->left);

    broadcast_expr *bcast = (broadcast_expr *) node;
    CSR_Matrix *Jx = node->left->jacobian;
    CSR_Matrix *J = node->jacobian;
    J->nnz = 0;

    if (bcast->type == BROADCAST_ROW)
    {
        for (int i = 0; i < node->d2; i++)
        {
            int nnz_in_row = Jx->p[i + 1] - Jx->p[i];
            tile_double(J->x + J->nnz, Jx->x + Jx->p[i], nnz_in_row, node->d1);
            J->nnz += nnz_in_row * node->d1;
        }
    }
    else if (bcast->type == BROADCAST_COL)
    {
        tile_double(J->x, Jx->x, Jx->nnz, node->d2);
    }
    else
    {
        tile_double(J->x, Jx->x, Jx->nnz, node->d1 * node->d2);
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    x->wsum_hess_init(x);

    /* Same sparsity as child - weights get summed */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (x->wsum_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));

    /* allocate space for weight vector */
    node->dwork = malloc(node->size * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    broadcast_expr *bcast = (broadcast_expr *) node;
    expr *x = node->left;

    /* Zero out the work array first */
    memset(node->dwork, 0, x->size * sizeof(double));

    if (bcast->type == BROADCAST_ROW)
    {
        /* (1, n) -> (m, n): each input element has m weights to sum */
        for (int j = 0; j < node->d2; j++)
        {
            for (int i = 0; i < node->d1; i++)
            {
                node->dwork[j] += w[i + j * node->d1];
            }
        }
    }
    else if (bcast->type == BROADCAST_COL)
    {
        /* (m, 1) -> (m, n): each input element has n weights to sum */
        for (int j = 0; j < node->d2; j++)
        {
            for (int i = 0; i < node->d1; i++)
            {
                node->dwork[i] += w[i + j * node->d1];
            }
        }
    }
    else
    {
        /* (1, 1) -> (m, n): scalar has m*n weights to sum */
        node->dwork[0] = 0.0;
        for (int k = 0; k < node->size; k++)
        {
            node->dwork[0] += w[k];
        }
    }

    x->eval_wsum_hess(x, node->dwork);
    memcpy(node->wsum_hess->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_broadcast(expr *child, int d1, int d2)
{
    // ---------------------------------------------------------------------------
    //                       determine broadcast type
    // ---------------------------------------------------------------------------
    broadcast_type type;

    if (child->d1 == 1 && child->d2 == d2)
    {
        type = BROADCAST_ROW;
    }
    else if (child->d1 == d1 && child->d2 == 1)
    {
        type = BROADCAST_COL;
    }
    else if (child->d1 == 1 && child->d2 == 1)
    {
        type = BROADCAST_SCALAR;
    }
    else
    {
        fprintf(
            stderr,
            "ERROR: inconsistency of broadcasting between DNLP-diff and CVXPY. \n");
        exit(1);
    }

    broadcast_expr *bcast = (broadcast_expr *) calloc(1, sizeof(broadcast_expr));
    expr *node = (expr *) bcast;

    // --------------------------------------------------------------------------
    //                  initialize the rest of the expression
    // --------------------------------------------------------------------------
    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);
    bcast->type = type;

    return node;
}

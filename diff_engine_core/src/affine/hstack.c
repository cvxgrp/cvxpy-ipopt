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

static void forward(expr *node, const double *u)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* children's forward passes */
    for (int i = 0; i < hnode->n_args; i++)
    {
        hnode->args[i]->forward(hnode->args[i], u);
    }

    /* concatenate values horizontally */
    int offset = 0;
    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        memcpy(node->value + offset, child->value, child->size * sizeof(double));
        offset += child->size;
    }
}

static void jacobian_init(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* initialize children's jacobians */
    int nnz = 0;
    for (int i = 0; i < hnode->n_args; i++)
    {
        assert(hnode->args[i] != NULL);
        hnode->args[i]->jacobian_init(hnode->args[i]);
        nnz += hnode->args[i]->jacobian->nnz;
    }

    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz);

    /* precompute sparsity pattern of this node's jacobian */
    int row_offset = 0;
    CSR_Matrix *A = node->jacobian;
    A->nnz = 0;

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        CSR_Matrix *B = child->jacobian;

        /* copy columns */
        memcpy(A->i + A->nnz, B->i, B->nnz * sizeof(int));

        /* set row pointers */
        for (int r = 0; r < child->size; r++)
        {
            A->p[row_offset + r] = A->nnz + B->p[r];
        }

        A->nnz += B->nnz;
        row_offset += child->size;
    }
    A->p[node->size] = A->nnz;
}

static void eval_jacobian(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    CSR_Matrix *A = node->jacobian;
    A->nnz = 0;

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        child->eval_jacobian(child);

        /* copy values */
        memcpy(A->x + A->nnz, child->jacobian->x,
               child->jacobian->nnz * sizeof(double));
        A->nnz += child->jacobian->nnz;
    }
}

static void wsum_hess_init(expr *node)
{
    /* initialize children's hessians */
    hstack_expr *hnode = (hstack_expr *) node;
    int nnz = 0;
    for (int i = 0; i < hnode->n_args; i++)
    {
        hnode->args[i]->wsum_hess_init(hnode->args[i]);
        nnz += hnode->args[i]->wsum_hess->nnz;
    }

    /* worst-case scenario the nnz of node->wsum_hess is the sum of children's
       nnz */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz);
    hnode->CSR_work = new_csr_matrix(node->n_vars, node->n_vars, nnz);

    /* fill sparsity pattern */
    CSR_Matrix *H = node->wsum_hess;
    H->nnz = 0;

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        copy_csr_matrix(H, hnode->CSR_work);
        sum_csr_matrices_fill_sparsity(hnode->CSR_work, child->wsum_hess, H);
    }
}

static void wsum_hess_eval(expr *node, const double *w)
{
    hstack_expr *hnode = (hstack_expr *) node;
    CSR_Matrix *H = node->wsum_hess;
    int row_offset = 0;
    memset(H->x, 0, H->nnz * sizeof(double));

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        child->eval_wsum_hess(child, w + row_offset);
        copy_csr_matrix(H, hnode->CSR_work);
        sum_csr_matrices_fill_values(hnode->CSR_work, child->wsum_hess, H);
        row_offset += child->size;
    }
}

static bool is_affine(const expr *node)
{
    const hstack_expr *hnode = (const hstack_expr *) node;

    for (int i = 0; i < hnode->n_args; i++)
    {
        if (!hnode->args[i]->is_affine(hnode->args[i]))
        {
            return false;
        }
    }
    return true;
}

static void free_type_data(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    for (int i = 0; i < hnode->n_args; i++)
    {
        free_expr(hnode->args[i]);
        hnode->args[i] = NULL;
    }

    free_csr_matrix(hnode->CSR_work);
    hnode->CSR_work = NULL;
    free(hnode->args);
    hnode->args = NULL;
}

expr *new_hstack(expr **args, int n_args, int n_vars)
{
    /* compute second dimension */
    int d2 = 0;
    for (int i = 0; i < n_args; i++)
    {
        assert(args[i]->d1 == args[0]->d1); /* all args must have same d1 */
        d2 += args[i]->d2;
    }

    /* Allocate the type-specific struct */
    hstack_expr *hnode = (hstack_expr *) calloc(1, sizeof(hstack_expr));
    expr *node = &hnode->base;
    init_expr(node, args[0]->d1, d2, n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, wsum_hess_eval, free_type_data);

    /* Set type-specific fields (deep copy args array) */
    hnode->args = (expr **) calloc(n_args, sizeof(expr *));
    hnode->n_args = n_args;
    for (int i = 0; i < n_args; i++)
    {
        hnode->args[i] = args[i];
        expr_retain(args[i]);
    }

    return node;
}

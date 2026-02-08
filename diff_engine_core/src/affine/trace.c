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
#include "utils/int_double_pair.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utils/iVec.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    x->forward(x, u);

    /* local forward pass */
    double sum = 0.0;
    int row_spacing = x->d1 + 1;
    for (int idx = 0; idx < x->size; idx += row_spacing)
    {
        sum += x->value[idx];
    }

    node->value[0] = sum;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    assert(x->d1 == x->d2);

    /* initialize child's jacobian */
    x->jacobian_init(x);

    // ---------------------------------------------------------------
    //    count total nnz and allocate matrix with sufficient space
    // ---------------------------------------------------------------
    const CSR_Matrix *A = x->jacobian;
    int total_nnz = 0;
    int row_spacing = x->d1 + 1;

    for (int row = 0; row < A->m; row += row_spacing)
    {
        total_nnz += A->p[row + 1] - A->p[row];
    }

    node->jacobian = new_csr_matrix(1, node->n_vars, total_nnz);

    // ---------------------------------------------------------------
    // fill sparsity pattern and idx_map
    // ---------------------------------------------------------------
    trace_expr *tnode = (trace_expr *) node;
    node->iwork = malloc(MAX(node->jacobian->n, total_nnz) * sizeof(int));

    /* the idx_map array maps each nonzero entry j in the original matrix A (from the
       selected, evenly spaced rows) to the corresponding index in the output row
       matrix C. Specifically, for each nonzero entry j in A (from the selected
       rows), idx_map[j] gives the position in C->x where the value from A->x[j]
       should be accumulated. */
    tnode->idx_map = malloc(x->jacobian->nnz * sizeof(int));
    sum_spaced_rows_into_row_csr_fill_sparsity_and_idx_map(
        A, node->jacobian, row_spacing, node->iwork, tnode->idx_map);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    trace_expr *tnode = (trace_expr *) node;

    /* evaluate child's jacobian */
    x->eval_jacobian(x);

    /* local jacobian */
    memset(node->jacobian->x, 0, node->jacobian->nnz * sizeof(double));
    idx_map_accumulator_with_spacing(x->jacobian, tnode->idx_map, node->jacobian->x,
                                     x->d1 + 1);
}

/* Placeholders for Hessian-related functions */
static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's hessian */
    x->wsum_hess_init(x);

    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    node->dwork = (double *) calloc(x->size, sizeof(double));

    /* We copy over the sparsity pattern from the child. This also includes the
       contribution to wsum_hess of entries of the child that will always have
       zero weight in eval_wsum_hess. We do this for simplicity. But the Hessian
       can for sure be made more sophisticated. */
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (x->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;

    int row_spacing = x->d1 + 1;
    for (int i = 0; i < x->size; i += row_spacing)
    {
        node->dwork[i] = w[0];
    }

    x->eval_wsum_hess(x, node->dwork);

    memcpy(node->wsum_hess->x, x->wsum_hess->x, sizeof(double) * x->wsum_hess->nnz);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    if (node)
    {
        trace_expr *tnode = (trace_expr *) node;
        free(tnode->idx_map);
    }
}

expr *new_trace(expr *child)
{
    trace_expr *tnode = (trace_expr *) calloc(1, sizeof(trace_expr));
    expr *node = &tnode->base;
    init_expr(node, 1, 1, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = child;
    expr_retain(child);

    return node;
}

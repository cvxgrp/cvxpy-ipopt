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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* negate values */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = -node->left->value[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* same sparsity pattern as child */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, x->jacobian->nnz);

    /* copy row pointers and column indices (sparsity pattern is constant) */
    memcpy(node->jacobian->p, x->jacobian->p, (x->jacobian->m + 1) * sizeof(int));
    memcpy(node->jacobian->i, x->jacobian->i, x->jacobian->nnz * sizeof(int));
}

static void eval_jacobian(expr *node)
{
    /* evaluate child's jacobian */
    node->left->eval_jacobian(node->left);

    /* negate values only (sparsity pattern set in jacobian_init) */
    CSR_Matrix *child_jac = node->left->jacobian;
    for (int k = 0; k < child_jac->nnz; k++)
    {
        node->jacobian->x[k] = -child_jac->x[k];
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    x->wsum_hess_init(x);

    /* same sparsity pattern as child */
    CSR_Matrix *child_hess = x->wsum_hess;
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, child_hess->nnz);

    /* copy row pointers and column indices (sparsity pattern is constant) */
    memcpy(node->wsum_hess->p, child_hess->p, (child_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, child_hess->i, child_hess->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* For f(x) = -g(x): d²f/dx² = -d²g/dx² */
    node->left->eval_wsum_hess(node->left, w);

    /* negate values (sparsity pattern set in wsum_hess_init) */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    for (int k = 0; k < child_hess->nnz; k++)
    {
        node->wsum_hess->x[k] = -child_hess->x[k];
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_neg(expr *child)
{
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, child->d1, child->d2, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}

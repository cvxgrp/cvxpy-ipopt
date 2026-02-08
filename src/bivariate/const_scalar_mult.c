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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Constant scalar multiplication: y = a * child where a is a constant double */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;

    /* child's forward pass */
    child->forward(child, u);

    /* local forward pass: multiply each element by scalar a */
    double a = ((const_scalar_mult_expr *) node)->a;
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = a * child->value[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child jacobian */
    x->jacobian_init(x);

    /* same sparsity as child */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, x->jacobian->nnz);
    memcpy(node->jacobian->p, x->jacobian->p, (node->size + 1) * sizeof(int));
    memcpy(node->jacobian->i, x->jacobian->i, x->jacobian->nnz * sizeof(int));
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    double a = ((const_scalar_mult_expr *) node)->a;

    /* evaluate child */
    child->eval_jacobian(child);

    /* scale child's jacobian */
    for (int j = 0; j < child->jacobian->nnz; j++)
    {
        node->jacobian->x[j] = a * child->jacobian->x[j];
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's weighted Hessian */
    x->wsum_hess_init(x);

    /* same sparsity as child */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (node->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    x->eval_wsum_hess(x, w);

    double a = ((const_scalar_mult_expr *) node)->a;
    for (int j = 0; j < x->wsum_hess->nnz; j++)
    {
        node->wsum_hess->x[j] = a * x->wsum_hess->x[j];
    }
}

static bool is_affine(const expr *node)
{
    /* Affine iff the child is affine */
    return node->left->is_affine(node->left);
}

expr *new_const_scalar_mult(double a, expr *child)
{
    const_scalar_mult_expr *mult_node =
        (const_scalar_mult_expr *) calloc(1, sizeof(const_scalar_mult_expr));
    expr *node = &mult_node->base;

    init_expr(node, child->d1, child->d2, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = child;
    mult_node->a = a;
    expr_retain(child);

    return node;
}

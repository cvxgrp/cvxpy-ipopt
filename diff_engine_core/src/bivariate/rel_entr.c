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
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------
// Implementation of relative entropy when both arguments are vectors.
// No chain rule is needed since both arguments must be variables.
// --------------------------------------------------------------------
static void forward_vector_args(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[i] * log(x->value[i] / y->value[i]);
    }
}

static void jacobian_init_vectors_args(expr *node)
{
    node->jacobian = new_csr_matrix(node->size, node->n_vars, 2 * node->size);

    expr *x = node->left;
    expr *y = node->right;
    assert(x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE);
    assert(x->var_id != y->var_id);

    /* if x has lower variable idx than y it should appear first */
    if (x->var_id < y->var_id)
    {
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->i[2 * j] = j + x->var_id;
            node->jacobian->i[2 * j + 1] = j + y->var_id;
            node->jacobian->p[j] = 2 * j;
        }
    }
    else
    {
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->i[2 * j] = j + y->var_id;
            node->jacobian->i[2 * j + 1] = j + x->var_id;
            node->jacobian->p[j] = 2 * j;
        }
    }

    node->jacobian->p[node->size] = 2 * node->size;
}

static void eval_jacobian_vector_args(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if x has lower variable idx than y */
    if (x->var_id < y->var_id)
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = log(x->value[i] / y->value[i]) + 1;
            node->jacobian->x[2 * i + 1] = -x->value[i] / y->value[i];
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = -x->value[i] / y->value[i];
            node->jacobian->x[2 * i + 1] = log(x->value[i] / y->value[i]) + 1;
        }
    }
}

static void wsum_hess_init_vector_args(expr *node)
{
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 4 * node->size);
    expr *x = node->left;
    expr *y = node->right;

    int i, var1_id, var2_id;

    if (x->var_id < y->var_id)
    {
        var1_id = x->var_id;
        var2_id = y->var_id;
    }
    else
    {
        var1_id = y->var_id;
        var2_id = x->var_id;
    }

    /* var1 rows of Hessian */
    for (i = 0; i < node->size; i++)
    {
        node->wsum_hess->p[var1_id + i] = 2 * i;
        node->wsum_hess->i[2 * i] = var1_id + i;
        node->wsum_hess->i[2 * i + 1] = var2_id + i;
    }

    int nnz = 2 * node->size;

    /* rows between var1 and var2 */
    for (i = var1_id + node->size; i < var2_id; i++)
    {
        node->wsum_hess->p[i] = nnz;
    }

    /* var2 rows of Hessian */
    for (i = 0; i < node->size; i++)
    {
        node->wsum_hess->p[var2_id + i] = nnz + 2 * i;
    }
    memcpy(node->wsum_hess->i + nnz, node->wsum_hess->i, nnz * sizeof(int));

    /* remaining rows */
    for (i = var2_id + node->size; i <= node->n_vars; i++)
    {
        node->wsum_hess->p[i] = 4 * node->size;
    }
}

static void eval_wsum_hess_vector_args(expr *node, const double *w)
{
    double *x = node->left->value;
    double *y = node->right->value;
    double *hess = node->wsum_hess->x;

    if (node->left->var_id < node->right->var_id)
    {
        for (int i = 0; i < node->size; i++)
        {
            hess[2 * i] = w[i] / x[i];
            hess[2 * i + 1] = -w[i] / y[i];
        }

        hess += 2 * node->size;

        for (int i = 0; i < node->size; i++)
        {
            hess[2 * i] = -w[i] / y[i];
            hess[2 * i + 1] = w[i] * x[i] / (y[i] * y[i]);
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            hess[2 * i] = w[i] * x[i] / (y[i] * y[i]);
            hess[2 * i + 1] = -w[i] / y[i];
        }

        hess += 2 * node->size;

        for (int i = 0; i < node->size; i++)
        {
            hess[2 * i] = -w[i] / y[i];
            hess[2 * i + 1] = w[i] / x[i];
        }
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_rel_entr_vector_args(expr *left, expr *right)
{
    /* if one argument is not a variable, we raise an error */
    if (left->var_id == NOT_A_VARIABLE || right->var_id == NOT_A_VARIABLE)
    {
        fprintf(stderr,
                "Error: Both arguments of relative entropy must be variables.\n");
        exit(EXIT_FAILURE);
    }

    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, left->d1, left->d2, left->n_vars, forward_vector_args,
              jacobian_init_vectors_args, eval_jacobian_vector_args, is_affine,
              wsum_hess_init_vector_args, eval_wsum_hess_vector_args, NULL);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    return node;
}

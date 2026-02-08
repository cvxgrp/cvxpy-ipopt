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
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------
// Implementation of relative entropy when first argument is a scalar
// and second argument is a vector.
// --------------------------------------------------------------------
static void forward_scalar_vector(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[0] * log(x->value[0] / y->value[i]);
    }
}

static void jacobian_init_scalar_vector(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    assert(x->d1 == 1 && x->d2 == 1);
    assert(x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE);
    assert(x->var_id != y->var_id);

    node->jacobian = new_csr_matrix(node->size, node->n_vars, 2 * node->size);

    if (x->var_id < y->var_id)
    {
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->i[2 * j] = x->var_id;
            node->jacobian->i[2 * j + 1] = y->var_id + j;
            node->jacobian->p[j] = 2 * j;
        }
    }
    else
    {
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->i[2 * j] = y->var_id + j;
            node->jacobian->i[2 * j + 1] = x->var_id;
            node->jacobian->p[j] = 2 * j;
        }
    }

    node->jacobian->p[node->size] = 2 * node->size;
}

static void eval_jacobian_scalar_vector(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    if (x->var_id < y->var_id)
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = log(x->value[0] / y->value[i]) + 1;
            node->jacobian->x[2 * i + 1] = -x->value[0] / y->value[i];
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = -x->value[0] / y->value[i];
            node->jacobian->x[2 * i + 1] = log(x->value[0] / y->value[i]) + 1;
        }
    }
}

static void wsum_hess_init_scalar_vector(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int var_id_x = x->var_id;
    int var_id_y = y->var_id;

    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 3 * node->size + 1);
    CSR_Matrix *H = node->wsum_hess;

    if (var_id_x < var_id_y)
    {
        H->p[var_id_x + 1] = node->size + 1;
        H->i[0] = var_id_x;
        for (int i = 0; i < node->size; i++)
        {
            H->i[i + 1] = var_id_y + i;
        }

        int offset = node->size + 1;
        for (int i = var_id_x + 1; i <= var_id_y; i++)
        {
            H->p[i] = offset;
        }

        for (int i = 0; i < node->size; i++)
        {
            H->p[var_id_y + i] = offset + 2 * i;
            H->i[offset + 2 * i] = var_id_x;
            H->i[offset + 2 * i + 1] = var_id_y + i;
        }

        for (int i = var_id_y + node->size; i <= node->n_vars; i++)
        {
            H->p[i] = 3 * node->size + 1;
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            H->p[var_id_y + i] = 2 * i;
            H->i[2 * i] = var_id_y + i;
            H->i[2 * i + 1] = var_id_x;
        }

        int offset = 2 * node->size;
        for (int i = var_id_y + node->size; i <= var_id_x; i++)
        {
            H->p[i] = offset;
        }

        H->p[var_id_x + 1] = offset + node->size + 1;
        for (int i = 0; i < node->size; i++)
        {
            H->i[offset + i] = var_id_y + i;
        }
        H->i[offset + node->size] = var_id_x;

        for (int i = var_id_x + 1; i <= node->n_vars; i++)
        {
            H->p[i] = 3 * node->size + 1;
        }
    }
}

static void eval_wsum_hess_scalar_vector(expr *node, const double *w)
{
    double x = node->left->value[0];
    double *y = node->right->value;
    double *H = node->wsum_hess->x;
    int var_id_x = node->left->var_id;
    int var_id_y = node->right->var_id;

    double diag_x = 0.0;
    for (int i = 0; i < node->size; i++)
    {
        diag_x += w[i] / x;
    }

    if (var_id_x < var_id_y)
    {
        H[0] = diag_x;
        for (int i = 0; i < node->size; i++)
        {
            H[i + 1] = -w[i] / y[i];
        }

        int offset = node->size + 1;
        for (int i = 0; i < node->size; i++)
        {
            H[offset + 2 * i] = -w[i] / y[i];
            H[offset + 2 * i + 1] = w[i] * x / (y[i] * y[i]);
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            H[2 * i] = w[i] * x / (y[i] * y[i]);
            H[2 * i + 1] = -w[i] / y[i];
        }

        int offset = 2 * node->size;
        for (int i = 0; i < node->size; i++)
        {
            H[offset + i] = -w[i] / y[i];
        }
        H[offset + node->size] = diag_x;
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_rel_entr_first_arg_scalar(expr *left, expr *right)
{
    assert(left->d1 == 1 && left->d2 == 1);
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, right->d1, right->d2, left->n_vars, forward_scalar_vector,
              jacobian_init_scalar_vector, eval_jacobian_scalar_vector, is_affine,
              wsum_hess_init_scalar_vector, eval_wsum_hess_scalar_vector, NULL);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    return node;
}

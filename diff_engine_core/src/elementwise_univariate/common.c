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
#include "elementwise_univariate.h"
#include "expr.h"
#include "subexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void jacobian_init_elementwise(expr *node)
{
    expr *child = node->left;

    /* if the variable is a child */
    if (child->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(node->size, node->n_vars, node->size);
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->p[j] = j;
            node->jacobian->i[j] = j + child->var_id;
        }
        node->jacobian->p[node->size] = node->size;
    }
    /* otherwise it will be a linear operator */
    else
    {
        CSR_Matrix *J = ((linear_op_expr *) child)->A_csr;
        node->jacobian = new_csr_matrix(J->m, J->n, J->nnz);

        node->dwork = (double *) malloc(node->size * sizeof(double));

        /* copy sparsity pattern of child */
        memcpy(node->jacobian->p, J->p, sizeof(int) * (J->m + 1));
        memcpy(node->jacobian->i, J->i, sizeof(int) * J->nnz);
    }
}

void eval_jacobian_elementwise(expr *node)
{
    expr *child = node->left;

    if (child->var_id != NOT_A_VARIABLE)
    {
        node->local_jacobian(node, node->jacobian->x);
    }
    else
    {
        /* Child will be a linear operator */
        linear_op_expr *lin_child = (linear_op_expr *) child;
        node->local_jacobian(node, node->dwork);
        diag_csr_mult_fill_values(node->dwork, lin_child->A_csr, node->jacobian);
    }
}

void wsum_hess_init_elementwise(expr *node)
{
    expr *child = node->left;
    int id = child->var_id;
    int i;

    /* if the variable is a child*/
    if (id != NOT_A_VARIABLE)
    {
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, node->size);

        for (i = 0; i < node->size; i++)
        {
            node->wsum_hess->p[id + i] = i;
            node->wsum_hess->i[i] = id + i;
        }

        for (i = id + node->size; i <= node->n_vars; i++)
        {
            node->wsum_hess->p[i] = node->size;
        }
    }
    /* otherwise it will be a linear operator */
    else
    {
        linear_op_expr *lin_child = (linear_op_expr *) child;
        node->wsum_hess = ATA_alloc(lin_child->A_csc);
    }
}

void eval_wsum_hess_elementwise(expr *node, const double *w)
{
    expr *child = node->left;

    if (child->var_id != NOT_A_VARIABLE)
    {
        node->local_wsum_hess(node, node->wsum_hess->x, w);
    }
    else
    {
        /* Child will be a linear operator */
        linear_op_expr *lin_child = (linear_op_expr *) child;
        node->local_wsum_hess(node, node->dwork, w);
        ATDA_fill_values(lin_child->A_csc, node->dwork, node->wsum_hess);
    }
}

bool is_affine_elementwise(const expr *node)
{
    (void) node;
    return false;
}

/* Helper function to initialize an already-allocated expr for elementwise operations
 * This is called when a power_expr or other type-specific struct is allocated
 * and we need to initialize the base expr fields
 */
void init_elementwise(expr *node, expr *child)
{
    /* Initialize base fields */
    init_expr(node, child->d1, child->d2, child->n_vars, NULL,
              jacobian_init_elementwise, eval_jacobian_elementwise,
              is_affine_elementwise, wsum_hess_init_elementwise,
              eval_wsum_hess_elementwise, NULL);

    /* Set left child */
    node->left = child;
    expr_retain(child);
}

expr *new_elementwise(expr *child)
{
    expr *node = (expr *) calloc(1, sizeof(expr));
    if (!node) return NULL;

    init_elementwise(node, child);
    return node;
}

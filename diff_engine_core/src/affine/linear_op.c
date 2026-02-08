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
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    linear_op_expr *lin_node = (linear_op_expr *) node;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A * x */
    csr_matvec(lin_node->A_csr, x->value, node->value, x->var_id);

    /* y += b (if offset exists) */
    if (lin_node->b != NULL)
    {
        for (int i = 0; i < node->size; i++)
        {
            node->value[i] += lin_node->b[i];
        }
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    linear_op_expr *lin_node = (linear_op_expr *) node;
    /* memory pointing to by A_csr will be freed when the jacobian is freed,
       so if the jacobian is not null we must not free A_csr. */

    if (!node->jacobian)
    {
        free_csr_matrix(lin_node->A_csr);
    }

    free_csc_matrix(lin_node->A_csc);

    if (lin_node->b != NULL)
    {
        free(lin_node->b);
        lin_node->b = NULL;
    }

    lin_node->A_csr = NULL;
    lin_node->A_csc = NULL;
}

static void jacobian_init(expr *node)
{
    node->jacobian = ((linear_op_expr *) node)->A_csr;
}

expr *new_linear(expr *u, const CSR_Matrix *A, const double *b)
{
    assert(u->d2 == 1);
    /* Allocate the type-specific struct */
    linear_op_expr *lin_node = (linear_op_expr *) calloc(1, sizeof(linear_op_expr));
    expr *node = &lin_node->base;
    init_expr(node, A->m, 1, u->n_vars, forward, jacobian_init, NULL, is_affine,
              NULL, NULL, free_type_data);
    node->left = u;
    expr_retain(u);

    /* Initialize type-specific fields */
    lin_node->A_csr = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, lin_node->A_csr);
    lin_node->A_csc = csr_to_csc(A);

    /* Initialize offset (copy b if provided, otherwise NULL) */
    if (b != NULL)
    {
        lin_node->b = (double *) malloc(A->m * sizeof(double));
        memcpy(lin_node->b, b, A->m * sizeof(double));
    }
    else
    {
        lin_node->b = NULL;
    }

    return node;
}

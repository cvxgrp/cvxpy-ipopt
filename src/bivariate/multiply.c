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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Implementation of elementwise multiplication when both arguments are vectors.
// If one argument is a scalar variable, the broadcasting should be represented
// as a linear operator child node? How to treat if one variable is a constant?
// ------------------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[i] * y->value[i];
    }
}

static void jacobian_init(expr *node)
{
    node->left->jacobian_init(node->left);
    node->right->jacobian_init(node->right);
    node->dwork = (double *) malloc(2 * node->size * sizeof(double));
    int nnz_max = node->left->jacobian->nnz + node->right->jacobian->nnz;
    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz_max);

    /* fill sparsity pattern */
    sum_csr_matrices_fill_sparsity(node->left->jacobian, node->right->jacobian,
                                   node->jacobian);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* chain rule */
    sum_scaled_csr_matrices_fill_values(x->jacobian, y->jacobian, node->jacobian,
                                        y->value, x->value);
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* both x and y are variables*/
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 2 * node->size);

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
            node->wsum_hess->p[var1_id + i] = i;
            node->wsum_hess->i[i] = var2_id + i;
        }

        int nnz = node->size;

        /* rows between var1 and var2 */
        for (i = var1_id + node->size; i < var2_id; i++)
        {
            node->wsum_hess->p[i] = nnz;
        }

        /* var2 rows of Hessian */
        for (i = 0; i < node->size; i++)
        {
            node->wsum_hess->p[var2_id + i] = nnz + i;
            node->wsum_hess->i[nnz + i] = var1_id + i;
        }

        /* remaining rows */
        nnz += node->size;
        for (i = var2_id + node->size; i <= node->n_vars; i++)
        {
            node->wsum_hess->p[i] = nnz;
        }
    }
    else
    {
        /* both are linear operators */
        CSC_Matrix *A = ((linear_op_expr *) x)->A_csc;
        CSC_Matrix *B = ((linear_op_expr *) y)->A_csc;

        /* Allocate workspace for Hessian computation */
        elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;
        CSR_Matrix *C; /* C = B^T diag(w) A */
        C = BTA_alloc(A, B);
        node->iwork = (int *) malloc(C->m * sizeof(int));

        CSR_Matrix *CT = AT_alloc(C, node->iwork);
        mul_node->CSR_work1 = C;
        mul_node->CSR_work2 = CT;

        /* Hessian is H = C + C^T where both are B->n x A->n, and can't be more than
         * 2 * nnz(C) */
        assert(C->m == node->n_vars && C->n == node->n_vars);
        node->wsum_hess = new_csr_matrix(C->m, C->n, 2 * C->nnz);

        /* fill sparsity pattern Hessian = C + C^T */
        sum_csr_matrices_fill_sparsity(C, CT, node->wsum_hess);
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    expr *y = node->right;

    /* both x and y are variables*/
    if (x->var_id != NOT_A_VARIABLE)
    {
        memcpy(node->wsum_hess->x, w, node->size * sizeof(double));
        memcpy(node->wsum_hess->x + node->size, w, node->size * sizeof(double));
    }
    else
    {
        /* both are linear operators */
        CSC_Matrix *A = ((linear_op_expr *) x)->A_csc;
        CSC_Matrix *B = ((linear_op_expr *) y)->A_csc;
        CSR_Matrix *C = ((elementwise_mult_expr *) node)->CSR_work1;
        CSR_Matrix *CT = ((elementwise_mult_expr *) node)->CSR_work2;

        /* Compute C = B^T diag(w) A */
        BTDA_fill_values(A, B, w, C);

        /* Compute CT = C^T = A^T diag(w) B */
        AT_fill_values(C, CT, node->iwork);

        /* Hessian = C + CT = B^T diag(w) A + A^T diag(w) B */
        sum_csr_matrices_fill_values(C, CT, node->wsum_hess);
    }
}

static void free_type_data(expr *node)
{
    free_csr_matrix(((elementwise_mult_expr *) node)->CSR_work1);
    free_csr_matrix(((elementwise_mult_expr *) node)->CSR_work2);
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_elementwise_mult(expr *left, expr *right)
{

    /* for correctness x and y must be (1) different variables, or (2) both must be
     * linear operators */
    if (left->var_id != NOT_A_VARIABLE && right->var_id != NOT_A_VARIABLE &&
        left->var_id == right->var_id)
    {
        fprintf(stderr, "Error: elementwise multiplication of a variable by itself "
                        "not supported.\n");
        exit(EXIT_FAILURE);
    }
    else if ((left->var_id != NOT_A_VARIABLE && right->var_id == NOT_A_VARIABLE) ||
             (left->var_id == NOT_A_VARIABLE && right->var_id != NOT_A_VARIABLE))
    {
        fprintf(stderr, "Error: elementwise multiplication of a variable by a "
                        "non-variable is not supported. (Both must be inserted "
                        "as linear operators)\n");
        exit(EXIT_FAILURE);
    }

    elementwise_mult_expr *mul_node =
        (elementwise_mult_expr *) calloc(1, sizeof(elementwise_mult_expr));
    expr *node = &mul_node->base;

    init_expr(node, left->d1, left->d2, left->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess,
              free_type_data);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);

    return node;
}

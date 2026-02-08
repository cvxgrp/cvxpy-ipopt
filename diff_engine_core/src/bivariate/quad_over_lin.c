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
#include "utils/CSC_Matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Implementation of quad-over-lin. The second argument will always be a variable
// that only appears in this node and as the left-hand side of another equality
// constraint.
// ------------------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    node->value[0] = 0.0;

    for (int i = 0; i < x->size; i++)
    {
        node->value[0] += x->value[i] * x->value[i];
    }

    node->value[0] /= y->value[0];
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if left node is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(1, node->n_vars, x->size + 1);
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = x->size + 1;

        /* if x has lower idx than y*/
        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < x->size; j++)
            {
                node->jacobian->i[j] = x->var_id + j;
            }
            node->jacobian->i[x->size] = y->var_id;
        }
        else /* y has lower idx than x */
        {
            node->jacobian->i[0] = y->var_id;
            for (int j = 0; j < x->size; j++)
            {
                node->jacobian->i[j + 1] = x->var_id + j;
            }
        }
    }
    else /* left node is not a variable (guaranteed to be a linear operator) */
    {
        linear_op_expr *lin_x = (linear_op_expr *) x;
        node->dwork = (double *) malloc(x->size * sizeof(double));

        /* compute required allocation and allocate jacobian */
        bool *col_nz = (bool *) calloc(
            node->n_vars, sizeof(bool)); /* TODO: could use iwork here instead*/
        int nonzero_cols = count_nonzero_cols(lin_x->A_csr, col_nz);
        node->jacobian = new_csr_matrix(1, node->n_vars, nonzero_cols + 1);

        /* precompute column indices */
        node->jacobian->nnz = 0;
        for (int j = 0; j < node->n_vars; j++)
        {
            if (col_nz[j])
            {
                node->jacobian->i[node->jacobian->nnz] = j;
                node->jacobian->nnz++;
            }
        }
        assert(nonzero_cols == node->jacobian->nnz);

        free(col_nz);

        /* insert y variable index at correct position */
        insert_idx(y->var_id, node->jacobian->i, node->jacobian->nnz);
        node->jacobian->nnz += 1;
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = node->jacobian->nnz;

        /* find position where y should be inserted */
        node->iwork = (int *) malloc(sizeof(int));
        for (int j = 0; j < node->jacobian->nnz; j++)
        {
            if (node->jacobian->i[j] == y->var_id)
            {
                node->iwork[0] = j;
                break;
            }
        }
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* if x has lower idx than y*/
        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < x->size; j++)
            {
                node->jacobian->x[j] = (2.0 * x->value[j]) / y->value[0];
            }
            node->jacobian->x[x->size] = -node->value[0] / y->value[0];
        }
        else /* y has lower idx than x */
        {
            node->jacobian->x[0] = -node->value[0] / y->value[0];
            for (int j = 0; j < x->size; j++)
            {
                node->jacobian->x[j + 1] = (2.0 * x->value[j]) / y->value[0];
            }
        }
    }
    else /* x is not a variable */
    {
        CSC_Matrix *A_csc = ((linear_op_expr *) x)->A_csc;

        /* local jacobian */
        for (int j = 0; j < x->size; j++)
        {
            node->dwork[j] = (2.0 * x->value[j]) / y->value[0];
        }

        /* chain rule (no derivative wrt y) using CSC format */
        csc_matvec_fill_values(A_csc, node->dwork, node->jacobian);

        /* insert derivative wrt y at right place (for correctness this assumes
           that y does not appear in the numerator, but this will always be
           the case since y is a new variable for the denominator */
        node->jacobian->x[node->iwork[0]] = -node->value[0] / y->value[0];
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int var_id_x = x->var_id;
    int var_id_y = y->var_id;

    /* if left node is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->wsum_hess =
            new_csr_matrix(node->n_vars, node->n_vars, 3 * x->size + 1);
        CSR_Matrix *H = node->wsum_hess;

        /* if x has lower idx than y*/
        if (var_id_x < var_id_y)
        {
            /* x rows: each row has 2 entries (diagonal element + element for y) */
            for (int i = 0; i < x->size; i++)
            {
                H->p[var_id_x + i] = 2 * i;
                H->i[2 * i] = var_id_x + i;
                H->i[2 * i + 1] = var_id_y;
            }

            /* rows between x and y are empty, all point to same offset */
            int offset = 2 * x->size;
            for (int i = var_id_x + x->size; i <= var_id_y; i++)
            {
                H->p[i] = offset;
            }

            /* y row has d1 + 1 entries */
            H->p[var_id_y + 1] = offset + x->size + 1;
            for (int i = 0; i < x->size; i++)
            {
                H->i[offset + i] = var_id_x + i;
            }
            H->i[offset + x->size] = var_id_y;

            /* remaining rows are empty */
            for (int i = var_id_y + 1; i <= node->n_vars; i++)
            {
                H->p[i] = 3 * x->size + 1;
            }
        }
        else /* y has lower idx than x */
        {
            /* y row has d1 + 1 entries */
            H->p[var_id_y + 1] = x->size + 1;
            H->i[0] = var_id_y;
            for (int i = 0; i < x->size; i++)
            {
                H->i[i + 1] = var_id_x + i;
            }

            /* rows between y and x are empty, all point to same offset */
            int offset = x->size + 1;
            for (int i = var_id_y + 1; i <= var_id_x; i++)
            {
                H->p[i] = offset;
            }

            /* x rows: each row has 2 entries */
            for (int i = 0; i < x->size; i++)
            {
                H->p[var_id_x + i] = offset + 2 * i;
                H->i[offset + 2 * i] = var_id_y;
                H->i[offset + 2 * i + 1] = var_id_x + i;
            }

            /* remaining rows are empty */
            for (int i = var_id_x + x->size; i <= node->n_vars; i++)
            {
                H->p[i] = 3 * x->size + 1;
            }
        }
    }
    else
    {
        /* TODO: implement */
        fprintf(stderr, "Error in quad_over_lin wsum_hess_init: non-variable child "
                        "not implemented\n");
        exit(1);
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    double y = node->right->value[0];
    double *H = node->wsum_hess->x;
    int var_id_x = node->left->var_id;
    int var_id_y = node->right->var_id;
    int x_size = node->left->size;
    double a = (2.0 * w[0]) / y;
    double b = -(2.0 * w[0]) / (y * y);

    /* if left node is a variable */
    if (var_id_x != NOT_A_VARIABLE)
    {
        /* if x has lower idx than y*/
        if (var_id_x < var_id_y)
        {
            /* x rows*/
            for (int i = 0; i < x_size; i++)
            {
                H[2 * i] = a;
                H[2 * i + 1] = b * x[i];
            }

            /* y row */
            int offset = 2 * x_size;
            for (int i = 0; i < x_size; i++)
            {
                H[offset + i] = b * x[i];
            }
            H[offset + x_size] = -b * node->value[0];
        }
        else /* y has lower idx than x */
        {
            /* y row */
            H[0] = -b * node->value[0];
            for (int i = 0; i < x_size; i++)
            {
                H[i + 1] = b * x[i];
            }

            /* x rows*/
            int offset = x_size + 1;
            for (int i = 0; i < x_size; i++)
            {
                H[offset + 2 * i] = b * x[i];
                H[offset + 2 * i + 1] = a;
            }
        }
    }
    else
    {
        /* TODO: implement */
        fprintf(stderr, "Error in quad_over_lin eval_wsum_hess: non-variable child "
                        "not implemented\n");
        exit(1);
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_quad_over_lin(expr *left, expr *right)
{
    /* if right is not a scalar or a variable by itself, we raise an error */
    if (right->var_id == NOT_A_VARIABLE && !(right->d1 == 1 && right->d2 == 1))
    {
        fprintf(stderr,
                "Error: Denominator of quad-over-lin must be a scalar variable.\n");
        exit(EXIT_FAILURE);
    }

    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, 1, 1, left->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    return node;
}

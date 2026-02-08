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

/* This file implement the atom 'left_matmul' corresponding to the operation y =
   A @ f(x), where A is a given matrix and f(x) is an arbitrary expression.
   Here, f(x) can be a vector-valued expression and a matrix-valued
   expression. The dimensions are A - m x n, f(x) - n x p, y - m x p.
   Note that here A does not have global column indices but it is a local matrix.
   This is an important distinction compared to linear_op_expr.

   * To compute the forward pass: vec(y) = A_kron @ vec(f(x)),
     where A_kron = I_p kron A is a Kronecker product of size (m*p) x (n*p),
     or more specificely, a block-diagonal matrix with p blocks of A along the
     diagonal.

   * To compute the Jacobian: J_y = A_kron @ J_f(x), where J_f(x) is the
     Jacobian of f(x) of size (n*p) x n_vars.

    * To compute the contribution to the Lagrange Hessian: we form
    w = A_kron^T @ lambda and then evaluate the hessian of f(x).

    Working in terms of A_kron unifies the implementation of f(x) being
    vector-valued or matrix-valued.


*/

#include "utils/utils.h"

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A_kron @ vec(f(x)) */
    csr_matvec_wo_offset(((left_matmul_expr *) node)->A, x->value, node->value);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    left_matmul_expr *lin_node = (left_matmul_expr *) node;
    free_csr_matrix(lin_node->A);
    free_csr_matrix(lin_node->AT);
    if (lin_node->CSC_work)
    {
        free_csc_matrix(lin_node->CSC_work);
    }
    lin_node->A = NULL;
    lin_node->AT = NULL;
    lin_node->CSC_work = NULL;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lin_node = (left_matmul_expr *) node;

    /* initialize child's jacobian and precompute sparsity of its transpose */
    x->jacobian_init(x);
    lin_node->CSC_work = csr_to_csc_fill_sparsity(x->jacobian, node->iwork);

    /* precompute sparsity of this node's jacobian */
    node->jacobian = csr_csc_matmul_alloc(lin_node->A, lin_node->CSC_work);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lin_node = (left_matmul_expr *) node;

    /* evaluate child's jacobian and convert to CSC */
    x->eval_jacobian(x);
    csr_to_csc_fill_values(x->jacobian, lin_node->CSC_work, node->iwork);

    /* compute this node's jacobian */
    csr_csc_matmul_fill_values(lin_node->A, lin_node->CSC_work, node->jacobian);
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's hessian */
    expr *x = node->left;
    x->wsum_hess_init(x);

    /* allocate this node's hessian with the same sparsity as child's */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (node->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));

    /* work for computing A^T w*/
    int A_n = ((left_matmul_expr *) node)->A->n;
    node->dwork = (double *) malloc(A_n * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* compute A^T w*/
    left_matmul_expr *lin_node = (left_matmul_expr *) node;
    csr_matvec_wo_offset(lin_node->AT, w, node->dwork);

    node->left->eval_wsum_hess(node->left, node->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

expr *new_left_matmul(expr *u, const CSR_Matrix *A)
{
    /* We expect u->d1 == A->n. However, numpy's broadcasting rules allow users to do
       A @ u where u is (n, ) which in C is actually (1, n). In that case the result
       of A @ u is (m, ), which is (1, m) according to broadcasting rules. We
       therefore check if this is the case. */
    int d1, d2, n_blocks;
    if (u->d1 == A->n)
    {
        d1 = A->m;
        d2 = u->d2;
        n_blocks = u->d2;
    }
    else if (u->d2 == A->n && u->d1 == 1)
    {
        d1 = 1;
        d2 = A->m;
        n_blocks = 1;
    }
    else
    {
        fprintf(stderr, "Error in new_left_matmul: dimension mismatch \n");
        exit(1);
    }

    /* Allocate the type-specific struct */
    left_matmul_expr *lin_node =
        (left_matmul_expr *) calloc(1, sizeof(left_matmul_expr));
    expr *node = &lin_node->base;
    init_expr(node, d1, d2, u->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = u;
    expr_retain(u);

    /* Initialize type-specific fields */
    lin_node->A = block_diag_repeat_csr(A, n_blocks);
    int alloc = MAX(lin_node->A->n, node->n_vars);
    node->iwork = (int *) malloc(alloc * sizeof(int));
    lin_node->AT = transpose(lin_node->A, node->iwork);
    lin_node->CSC_work = NULL;

    return node;
}

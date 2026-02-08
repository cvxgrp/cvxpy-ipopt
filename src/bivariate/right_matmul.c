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
#include <stdlib.h>

/* This file implements the atom 'right_matmul' corresponding to the operation y =
   f(x) @ A, where A is a given matrix and f(x) is an arbitrary expression.
   Here, f(x) can be a vector-valued expression and a matrix-valued
   expression. The dimensions are f(x) - p x n, A - n x q, y - p x q.
   Note that here A does not have global column indices but it is a local matrix.
   This is an important distinction compared to linear_op_expr.

   * To compute the forward pass: vec(y) = B @ vec(f(x)),
     where B = A^T kron I_p is a Kronecker product of size (p*q) x (p*n).

   * To compute the Jacobian: J_y = B @ J_f(x), where J_f(x) is the
     Jacobian of f(x) of size (p*n) x n_vars.

    * To compute the contribution to the Lagrange Hessian: we form
    w_child = B^T @ w and then evaluate the hessian of f(x).
*/

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = x * A, vec(y) = B @ vec(x) */
    csr_matvec_wo_offset(((right_matmul_expr *) node)->B, x->value, node->value);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    right_matmul_expr *right_node = (right_matmul_expr *) node;
    free_csr_matrix(right_node->B);
    free_csr_matrix(right_node->BT);
    if (right_node->CSC_work)
    {
        free_csc_matrix(right_node->CSC_work);
    }
    right_node->B = NULL;
    right_node->BT = NULL;
    right_node->CSC_work = NULL;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    right_matmul_expr *right_node = (right_matmul_expr *) node;

    /* initialize child's jacobian and precompute sparsity of its transpose */
    x->jacobian_init(x);
    right_node->CSC_work = csr_to_csc_fill_sparsity(x->jacobian, node->iwork);

    /* precompute sparsity of this node's jacobian */
    node->jacobian = csr_csc_matmul_alloc(right_node->B, right_node->CSC_work);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    right_matmul_expr *right_node = (right_matmul_expr *) node;

    /* evaluate child's jacobian and convert to CSC*/
    x->eval_jacobian(x);
    csr_to_csc_fill_values(x->jacobian, right_node->CSC_work, node->iwork);

    /* compute this node's jacobian */
    csr_csc_matmul_fill_values(right_node->B, right_node->CSC_work, node->jacobian);
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

    /* Allocate workspace for B^T @ w */
    node->dwork = (double *) malloc(x->d1 * x->d2 * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Compute B^T @ w, where B = A^T ⊗ I_p */
    right_matmul_expr *right_node = (right_matmul_expr *) node;

    /* B^T @ w computes the weights for the child expression */
    csr_matvec_wo_offset(right_node->BT, w, node->dwork);

    /* Propagate to child */
    node->left->eval_wsum_hess(node->left, node->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

expr *new_right_matmul(expr *u, const CSR_Matrix *A)
{
    /* Allocate the type-specific struct */
    right_matmul_expr *right_matmul_node =
        (right_matmul_expr *) calloc(1, sizeof(right_matmul_expr));
    expr *node = &right_matmul_node->base;

    /* Output dimensions: u is p x n, A is n x q, output is p x q */
    int p = u->d1;
    int q = A->n;

    init_expr(node, p, q, u->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = u;
    expr_retain(u);

    /* create B = A^T kron I_p and its transpose */
    node->iwork = (int *) malloc(node->n_vars * sizeof(int));
    CSR_Matrix *AT = transpose(A, node->iwork);
    right_matmul_node->B = kron_identity_csr(AT, p);
    right_matmul_node->BT = kron_identity_csr(A, p);
    free_csr_matrix(AT);

    right_matmul_node->CSC_work = NULL;

    return node;
}

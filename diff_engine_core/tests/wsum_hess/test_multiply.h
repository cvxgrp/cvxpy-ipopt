#include "affine.h"
#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_multiply_1()
{
    // Total 12 variables: [?, ?, ?, x0, x1, x2, ?, ?, y0, y1, y2, ?]
    // x has var_id = 3, y has var_id = 8
    // x = [1, 2, 3], y = [4, 5, 6]
    // w = [1, 2, 3]
    // Hessian structure: [0, diag(w); diag(w), 0]

    double u_vals[12] = {0, 0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0, 0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 3, 12);
    expr *y = new_variable(3, 1, 8, 12);
    expr *node = new_elementwise_mult(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, w);

    int expected_p[13] = {0, 0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6};
    int expected_i[6] = {8, 9, 10, 3, 4, 5};
    double expected_x[6] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 13));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 6));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 6));

    free_expr(node);
    return 0;
}

const char *test_wsum_hess_multiply_sparse_random()
{
    /* Test with larger random sparse matrices
     * A: 5x10 CSR matrix
     * B: 5x10 CSR matrix
     * x: 10-dimensional variable with var_id = 0, n_vars = 10
     * Expected Hessian: 10x10 sparse matrix
     */

    /* Create A matrix (5x10) */
    CSR_Matrix *A = new_csr_matrix(5, 10, 10);
    double Ax[10] = {-1.44165273, -1.13687223, 0.55892257,  0.24912193,  0.84959744,
                     -0.23998915, 0.5913356,   -1.21627912, -0.50379166, 0.41531801};
    int Ai[10] = {1, 2, 4, 8, 2, 3, 8, 9, 1, 2};
    int Ap[6] = {0, 3, 3, 4, 8, 10};
    memcpy(A->x, Ax, 10 * sizeof(double));
    memcpy(A->i, Ai, 10 * sizeof(int));
    memcpy(A->p, Ap, 6 * sizeof(int));

    /* Create B matrix (5x10) */
    CSR_Matrix *B = new_csr_matrix(5, 10, 10);
    double Bx[10] = {1.27549062,  0.04194731, -0.4356034,  0.405574,   1.34670487,
                     -0.57738638, 0.9411464,  -0.31563179, 1.90831766, -0.89802958};
    int Bi[10] = {0, 3, 5, 7, 0, 5, 0, 3, 7, 9};
    int Bp[6] = {0, 1, 4, 6, 8, 10};
    memcpy(B->x, Bx, 10 * sizeof(double));
    memcpy(B->i, Bi, 10 * sizeof(int));
    memcpy(B->p, Bp, 6 * sizeof(int));

    /* Create variable and linear operators */
    expr *x = new_variable(10, 1, 0, 10);
    expr *Ax_node = new_linear(x, A, NULL);
    expr *Bx_node = new_linear(x, B, NULL);

    /* Create multiply node */
    expr *mult_node = new_elementwise_mult(Ax_node, Bx_node);

    /* Forward pass with unit input */
    double u_vals[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    mult_node->forward(mult_node, u_vals);

    /* Initialize and evaluate Hessian */
    mult_node->wsum_hess_init(mult_node);
    double w[5] = {0.50646339, 0.44756224, 0.67295241, 0.16424956, 0.03031469};
    mult_node->eval_wsum_hess(mult_node, w);

    /* Expected Hessian in CSR format (10x10) */
    int expected_p[11] = {0, 6, 9, 13, 18, 19, 20, 20, 22, 25, 29};
    int expected_i[29] = {1, 2, 3, 4, 8, 9, 0, 7, 9, 0, 3, 7, 9, 0, 2,
                          3, 8, 9, 0, 8, 1, 2, 0, 3, 5, 0, 1, 2, 3};
    double expected_x[29] = {
        -0.93129225, -0.60307409, -0.03709821, 0.361058,    0.31718166,  -0.18801593,
        -0.93129225, -0.02914438, 0.01371497,  -0.60307409, -0.04404516, 0.02402617,
        -0.01130641, -0.03709821, -0.04404516, 0.02488322,  -0.03065625, 0.06305481,
        0.361058,    -0.09679721, -0.02914438, 0.02402617,  0.31718166,  -0.03065625,
        -0.09679721, -0.18801593, 0.01371497,  -0.01130641, 0.06305481};

    mu_assert("p array fails",
              cmp_int_array(mult_node->wsum_hess->p, expected_p, 11));
    mu_assert("i array fails",
              cmp_int_array(mult_node->wsum_hess->i, expected_i, 29));
    mu_assert("x array fails",
              cmp_double_array(mult_node->wsum_hess->x, expected_x, 29));

    /* Cleanup */
    free_expr(mult_node);
    free_csr_matrix(A);
    free_csr_matrix(B);

    return 0;
}

const char *test_wsum_hess_multiply_linear_ops()
{
    /* Test Hessian for mult(Ax, Bx) where A, B are 4x3 linear operators
     * A = [[1.0, 0.0, 2.0],
     *      [0.0, 3.0, 0.0],
     *      [4.0, 0.0, 5.0],
     *      [0.0, 6.0, 0.0]]
     * B = [[1.0, 0.0, 4.0],
     *      [0.0, 2.0, 7.0],
     *      [3.0, 0.0, 2.0],
     *      [0.0, 4.0, -1.0]]
     * x has var_id = 0, n_vars = 3
     * w = [1.0, 2.0, 3.0, 4.0]
     * Hessian = A^T diag(w) B + B^T diag(w) A
     * Expected (from numpy):
     * [[ 74.   0.  75.]
     *  [  0. 216.  18.]
     *  [ 75.  18.  76.]]
     */

    /* Create CSR matrix A */
    CSR_Matrix *A = new_csr_matrix(4, 3, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 2, 1, 0, 2, 1};
    int Ap[5] = {0, 2, 3, 5, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    /* Create CSR matrix B */
    CSR_Matrix *B = new_csr_matrix(4, 3, 8);
    double Bx[8] = {1.0, 4.0, 2.0, 7.0, 3.0, 2.0, 4.0, -1.0};
    int Bi[8] = {0, 2, 1, 2, 0, 2, 1, 2};
    int Bp[5] = {0, 2, 4, 6, 8};
    memcpy(B->x, Bx, 8 * sizeof(double));
    memcpy(B->i, Bi, 8 * sizeof(int));
    memcpy(B->p, Bp, 5 * sizeof(int));

    /* Create linear operator expressions */
    expr *x = new_variable(3, 1, 0, 3);
    expr *Ax_node = new_linear(x, A, NULL);
    expr *Bx_node = new_linear(x, B, NULL);

    /* Create elementwise multiply node */
    expr *mult_node = new_elementwise_mult(Ax_node, Bx_node);

    /* Forward pass */
    double u_vals[3] = {1.0, 1.0, 1.0};
    mult_node->forward(mult_node, u_vals);

    /* Initialize Hessian structure */
    mult_node->wsum_hess_init(mult_node);

    /* Evaluate Hessian with weights */
    double w[4] = {1.0, 2.0, 3.0, 4.0};
    mult_node->eval_wsum_hess(mult_node, w);

    /* Check sparsity pattern and values */
    /* Expected CSR format:
     * indptr: [0, 2, 4, 7]
     * indices: [0, 2, 1, 2, 0, 1, 2]
     * data: [74.0, 75.0, 216.0, 18.0, 75.0, 18.0, 76.0]
     */
    int expected_p[4] = {0, 2, 4, 7};
    int expected_i[7] = {0, 2, 1, 2, 0, 1, 2};
    double expected_x[7] = {74.0, 75.0, 216.0, 18.0, 75.0, 18.0, 76.0};

    mu_assert("p array fails",
              cmp_int_array(mult_node->wsum_hess->p, expected_p, 4));
    mu_assert("i array fails",
              cmp_int_array(mult_node->wsum_hess->i, expected_i, 7));
    mu_assert("x array fails",
              cmp_double_array(mult_node->wsum_hess->x, expected_x, 7));

    /* Cleanup */
    free_expr(mult_node);
    free_csr_matrix(A);
    free_csr_matrix(B);

    return 0;
}

const char *test_wsum_hess_multiply_2()
{
    // Total 12 variables: [?, ?, ?, y0, y1, y2, ?, ?, x0, x1, x2, ?]
    // y has var_id = 3, x has var_id = 8
    // x = [1, 2, 3], y = [4, 5, 6]
    // w = [1, 2, 3]
    // Hessian structure: [0, diag(w); diag(w), 0]
    // Should get the same result as test 1

    double u_vals[12] = {0, 0, 0, 4.0, 5.0, 6.0, 0, 0, 1.0, 2.0, 3.0, 0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 8, 12);
    expr *y = new_variable(3, 1, 3, 12);
    expr *node = new_elementwise_mult(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, w);

    int expected_p[13] = {0, 0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6};
    int expected_i[6] = {8, 9, 10, 3, 4, 5};
    double expected_x[6] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 13));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 6));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 6));

    free_expr(node);
    return 0;
}

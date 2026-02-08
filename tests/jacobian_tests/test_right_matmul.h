#include <math.h>
#include <stdio.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_right_matmul_log()
{
    /* Test Jacobian of log(x) @ A where:
     * x is 2x2 variable at x = [[1, 2], [3, 4]] (vectorized column-wise: [1, 3, 2,
     * 4]) A is 2x3 sparse matrix [1, 0, 2; 3, 0, 4] Output: log(x) @ A is 2x3
     */
    double x_vals[4] = {1.0, 3.0, 2.0, 4.0}; // column-wise vectorization
    expr *x = new_variable(2, 2, 0, 4);

    /* Create sparse matrix A in CSR format (2x3) */
    CSR_Matrix *A = new_csr_matrix(2, 3, 4);
    int A_p[3] = {0, 2, 4};
    int A_i[4] = {0, 2, 0, 2};
    double A_x[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(A->p, A_p, 3 * sizeof(int));
    memcpy(A->i, A_i, 4 * sizeof(int));
    memcpy(A->x, A_x, 4 * sizeof(double));

    expr *log_x = new_log(x);
    expr *log_x_A = new_right_matmul(log_x, A);

    log_x_A->forward(log_x_A, x_vals);
    log_x_A->jacobian_init(log_x_A);
    log_x_A->eval_jacobian(log_x_A);

    /* Expected jacobian values */
    double expected_Ax[8] = {
        1.0,       /* row 0, col 0 */
        1.5,       /* row 0, col 2 */
        1.0 / 3.0, /* row 1, col 1 */
        0.75,      /* row 1, col 3 */
        2.0,       /* row 4, col 0 */
        2.0,       /* row 4, col 2 */
        2.0 / 3.0, /* row 5, col 1 */
        1.0        /* row 5, col 3 */
    };
    int expected_Ai[8] = {0, 2, 1, 3, 0, 2, 1, 3};
    int expected_Ap[7] = {0, 2, 4, 4, 4, 6, 8};

    mu_assert("vals fail", cmp_double_array(log_x_A->jacobian->x, expected_Ax, 8));
    mu_assert("cols fail", cmp_int_array(log_x_A->jacobian->i, expected_Ai, 8));
    mu_assert("rows fail", cmp_int_array(log_x_A->jacobian->p, expected_Ap, 7));

    free_csr_matrix(A);
    free_expr(log_x_A);
    return 0;
}

const char *test_jacobian_right_matmul_log_vector()
{
    /* Test Jacobian of log(x) @ A where:
     * x is 1x3 variable at x = [1, 2, 3]
     * A is 3x2 sparse matrix [[1, 0], [2, 3], [0, 4]]
     * Output: log(x) @ A is 1x2
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(1, 3, 0, 3);

    /* Create sparse matrix A in CSR format (3x2) */
    CSR_Matrix *A = new_csr_matrix(3, 2, 4);
    int A_p[4] = {0, 1, 3, 4};
    int A_i[4] = {0, 0, 1, 1};
    double A_x[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(A->p, A_p, 4 * sizeof(int));
    memcpy(A->i, A_i, 4 * sizeof(int));
    memcpy(A->x, A_x, 4 * sizeof(double));

    expr *log_x = new_log(x);
    expr *log_x_A = new_right_matmul(log_x, A);

    log_x_A->forward(log_x_A, x_vals);
    log_x_A->jacobian_init(log_x_A);
    log_x_A->eval_jacobian(log_x_A);

    /* Expected jacobian values: A^T @ diag(1/x) */
    double expected_Ax[4] = {
        1.0,      /* row 0, col 0 */
        1.0,      /* row 0, col 1 */
        1.5,      /* row 1, col 1 */
        4.0 / 3.0 /* row 1, col 2 */
    };
    int expected_Ai[4] = {0, 1, 1, 2};
    int expected_Ap[3] = {0, 2, 4};

    mu_assert("vals fail", cmp_double_array(log_x_A->jacobian->x, expected_Ax, 4));
    mu_assert("cols fail", cmp_int_array(log_x_A->jacobian->i, expected_Ai, 4));
    mu_assert("rows fail", cmp_int_array(log_x_A->jacobian->p, expected_Ap, 3));

    free_csr_matrix(A);
    free_expr(log_x_A);
    return 0;
}

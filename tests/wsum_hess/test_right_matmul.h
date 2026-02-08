#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_right_matmul()
{
    /* Test weighted sum of Hessian of log(x) @ A where:
     * x is 2x2 variable at x = [[1, 2], [3, 4]] (vectorized column-wise: [1, 3, 2,
     * 4]) A is 2x3 sparse matrix [[1, 0, 2], [3, 0, 4]] Output: log(x) @ A is 2x3
     * Weights w is 2x3 (vectorized column-wise: w = [w00, w10, w01, w11, w02, w12])
     *   = [1, 2, 3, 4, 5, 6]
     */
    double x_vals[4] = {1.0, 3.0, 2.0, 4.0};      // column-wise vectorization
    double w[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // weights for 2x3 output

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
    log_x_A->wsum_hess_init(log_x_A);
    log_x_A->eval_wsum_hess(log_x_A, w);

    /* Expected wsum_hess: diagonal matrix with 4 entries */
    double expected_x[4] = {
        -11.0,       /* position [0,0]: -11 */
        -14.0 / 9.0, /* position [1,1]: -14/9 */
        -23.0 / 4.0, /* position [2,2]: -23/4 */
        -15.0 / 8.0  /* position [3,3]: -15/8 */
    };
    int expected_i[4] = {0, 1, 2, 3};
    int expected_p[5] = {0, 1, 2, 3, 4}; /* each row has 1 diagonal entry */

    mu_assert("vals incorrect",
              cmp_double_array(log_x_A->wsum_hess->x, expected_x, 4));
    mu_assert("cols incorrect", cmp_int_array(log_x_A->wsum_hess->i, expected_i, 4));
    mu_assert("rows incorrect", cmp_int_array(log_x_A->wsum_hess->p, expected_p, 5));

    free_csr_matrix(A);
    free_expr(log_x_A);
    return 0;
}

const char *test_wsum_hess_right_matmul_vector()
{
    /* Test weighted sum of Hessian of log(x) @ A where:
     * x is 1x3 variable at x = [1, 2, 3]
     * A is 3x2 sparse matrix [[1, 0], [2, 3], [0, 4]]
     * Output: log(x) @ A is 1x2
     * Weights w is 1x2 (vectorized: w = [w0, w1] = [1, 2])
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    double w[2] = {1.0, 2.0}; // weights for 1x2 output

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
    log_x_A->wsum_hess_init(log_x_A);
    log_x_A->eval_wsum_hess(log_x_A, w);

    /* Expected wsum_hess: diagonal matrix with 3 entries */
    double expected_x[3] = {
        -1.0,      /* position [0,0]: -1 */
        -2.0,      /* position [1,1]: -2 */
        -8.0 / 9.0 /* position [2,2]: -8/9 */
    };
    int expected_i[3] = {0, 1, 2};
    int expected_p[4] = {0, 1, 2, 3}; /* each row has 1 diagonal entry */

    mu_assert("vals incorrect",
              cmp_double_array(log_x_A->wsum_hess->x, expected_x, 3));
    mu_assert("cols incorrect", cmp_int_array(log_x_A->wsum_hess->i, expected_i, 3));
    mu_assert("rows incorrect", cmp_int_array(log_x_A->wsum_hess->p, expected_p, 4));

    free_csr_matrix(A);
    free_expr(log_x_A);
    return 0;
}

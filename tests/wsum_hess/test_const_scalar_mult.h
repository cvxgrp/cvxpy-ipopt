#include <math.h>
#include <stdio.h>

#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

/* Test: y = a * log(x) where a is a scalar constant */

const char *test_wsum_hess_const_scalar_mult_log_vector()
{
    /* Create variable x: [1.0, 2.0, 4.0] */
    double u_vals[3] = {1.0, 2.0, 4.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create scalar mult node: y = 2.5 * log(x) */
    double a = 2.5;
    expr *y = new_const_scalar_mult(a, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate weighted Hessian with w = [1.0, 0.5, 0.25] */
    y->wsum_hess_init(y);
    double w[3] = {1.0, 0.5, 0.25};
    y->eval_wsum_hess(y, w);

    /* For y = a * log(x), the Hessian is:
     * H = a * H_log = a * diag([-1/x_i^2])
     * With weights w and scalar a:
     * H_weighted = a * diag([-w_i / x_i^2])
     *
     * Expected diagonal: 2.5 * [-1/1^2, -0.5/2^2, -0.25/4^2]
     *                  = 2.5 * [-1, -0.125, -0.015625]
     *                  = [-2.5, -0.3125, -0.0390625]
     */
    double expected_x[3] = {-2.5, -0.3125, -0.0390625};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("scalar mult log hess: x values fail",
              cmp_double_array(y->wsum_hess->x, expected_x, 3));
    mu_assert("scalar mult log hess: row pointers fail",
              cmp_int_array(y->wsum_hess->p, expected_p, 4));
    mu_assert("scalar mult log hess: column indices fail",
              cmp_int_array(y->wsum_hess->i, expected_i, 3));

    free_expr(y);
    return 0;
}

const char *test_wsum_hess_const_scalar_mult_log_matrix()
{
    /* Create variable x as 2x2 matrix: [[1.0, 2.0], [4.0, 8.0]] */
    double u_vals[4] = {1.0, 2.0, 4.0, 8.0};
    expr *x = new_variable(2, 2, 0, 4);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create scalar mult node: y = 3.0 * log(x) */
    double a = 3.0;
    expr *y = new_const_scalar_mult(a, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate weighted Hessian with w = [1.0, 1.0, 1.0, 1.0] */
    y->wsum_hess_init(y);
    double w[4] = {1.0, 1.0, 1.0, 1.0};
    y->eval_wsum_hess(y, w);

    /* Expected diagonal: 3.0 * [-1/1^2, -1/2^2, -1/4^2, -1/8^2]
     *                  = 3.0 * [-1, -0.25, -0.0625, -0.015625]
     *                  = [-3.0, -0.75, -0.1875, -0.046875]
     */
    double expected_x[4] = {-3.0, -0.75, -0.1875, -0.046875};
    int expected_p[5] = {0, 1, 2, 3, 4};
    int expected_i[4] = {0, 1, 2, 3};

    mu_assert("scalar mult log hess matrix: x values fail",
              cmp_double_array(y->wsum_hess->x, expected_x, 4));
    mu_assert("scalar mult log hess matrix: row pointers fail",
              cmp_int_array(y->wsum_hess->p, expected_p, 5));
    mu_assert("scalar mult log hess matrix: column indices fail",
              cmp_int_array(y->wsum_hess->i, expected_i, 4));

    free_expr(y);
    return 0;
}

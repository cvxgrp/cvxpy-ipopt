#include <stdio.h>

#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

/* Test: y = a * log(x) where a is a scalar constant */

const char *test_jacobian_const_scalar_mult_log_vector()
{
    /* Create variable x: [1.0, 2.0, 4.0] with 3 elements */
    double u_vals[3] = {1.0, 2.0, 4.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create scalar mult node: y = 2.5 * log(x) */
    double a = 2.5;
    expr *y = new_const_scalar_mult(a, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate jacobian */
    y->jacobian_init(y);
    y->eval_jacobian(y);

    /* Expected jacobian: 2.5 * [1/1, 1/2, 1/4] = [2.5, 1.25, 0.625] */
    double expected_x[3] = {2.5, 1.25, 0.625};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("scalar mult log: x values fail",
              cmp_double_array(y->jacobian->x, expected_x, 3));
    mu_assert("scalar mult log: row pointers fail",
              cmp_int_array(y->jacobian->p, expected_p, 4));
    mu_assert("scalar mult log: column indices fail",
              cmp_int_array(y->jacobian->i, expected_i, 3));

    free_expr(y);
    return 0;
}

const char *test_jacobian_const_scalar_mult_log_matrix()
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

    /* Initialize and evaluate jacobian */
    y->jacobian_init(y);
    y->eval_jacobian(y);

    /* Expected jacobian: 3.0 * [1/1, 1/2, 1/4, 1/8] = [3.0, 1.5, 0.75, 0.375] */
    double expected_x[4] = {3.0, 1.5, 0.75, 0.375};
    int expected_p[5] = {0, 1, 2, 3, 4};
    int expected_i[4] = {0, 1, 2, 3};

    mu_assert("scalar mult log matrix: x values fail",
              cmp_double_array(y->jacobian->x, expected_x, 4));
    mu_assert("scalar mult log matrix: row pointers fail",
              cmp_int_array(y->jacobian->p, expected_p, 5));
    mu_assert("scalar mult log matrix: column indices fail",
              cmp_int_array(y->jacobian->i, expected_i, 4));

    free_expr(y);
    return 0;
}

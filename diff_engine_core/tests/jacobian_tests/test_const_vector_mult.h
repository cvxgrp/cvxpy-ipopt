#include <stdio.h>

#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

/* Test: y = a ∘ log(x) where a is a constant vector */

const char *test_jacobian_const_vector_mult_log_vector()
{
    /* Create variable x: [1.0, 2.0, 4.0] */
    double u_vals[3] = {1.0, 2.0, 4.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create vector mult node: y = [2.0, 3.0, 4.0] ∘ log(x) */
    double a[3] = {2.0, 3.0, 4.0};
    expr *y = new_const_vector_mult(a, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate jacobian */
    y->jacobian_init(y);
    y->eval_jacobian(y);

    /* Expected jacobian (row-wise scaling):
     * Row 0: 2.0 * [1/1] = [2.0]
     * Row 1: 3.0 * [1/2] = [1.5]
     * Row 2: 4.0 * [1/4] = [1.0]
     */
    double expected_x[3] = {2.0, 1.5, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vector mult log: x values fail",
              cmp_double_array(y->jacobian->x, expected_x, 3));
    mu_assert("vector mult log: row pointers fail",
              cmp_int_array(y->jacobian->p, expected_p, 4));
    mu_assert("vector mult log: column indices fail",
              cmp_int_array(y->jacobian->i, expected_i, 3));

    free_expr(y);
    return 0;
}

const char *test_jacobian_const_vector_mult_log_matrix()
{
    /* Create variable x as 2x2 matrix: [[1.0, 2.0], [4.0, 8.0]] */
    double u_vals[4] = {1.0, 2.0, 4.0, 8.0};
    expr *x = new_variable(2, 2, 0, 4);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create vector mult node: y = [1.5, 2.5, 3.5, 4.5] ∘ log(x) */
    double a[4] = {1.5, 2.5, 3.5, 4.5};
    expr *y = new_const_vector_mult(a, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate jacobian */
    y->jacobian_init(y);
    y->eval_jacobian(y);

    /* Expected jacobian (row-wise scaling):
     * Row 0: 1.5 * [1/1] = [1.5]
     * Row 1: 2.5 * [1/2] = [1.25]
     * Row 2: 3.5 * [1/4] = [0.875]
     * Row 3: 4.5 * [1/8] = [0.5625]
     */
    double expected_x[4] = {1.5, 1.25, 0.875, 0.5625};
    int expected_p[5] = {0, 1, 2, 3, 4};
    int expected_i[4] = {0, 1, 2, 3};

    mu_assert("vector mult log matrix: x values fail",
              cmp_double_array(y->jacobian->x, expected_x, 4));
    mu_assert("vector mult log matrix: row pointers fail",
              cmp_int_array(y->jacobian->p, expected_p, 5));
    mu_assert("vector mult log matrix: column indices fail",
              cmp_int_array(y->jacobian->i, expected_i, 4));

    free_expr(y);
    return 0;
}

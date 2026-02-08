#include <stdio.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_sum_log()
{
    /* Test Jacobian of sum(log(x)) where x is 3x1 variable evaluated at [1, 2, 3]
     * Jacobian should be [0, 0, 1/1, 1/2, 1/3, 0] = [0, 0, 1.0, 0.5, 1/3, 0] as a
     * 1x6 sparse matrix
     */
    double u_vals[5] = {0.0, 0.0, 1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 2, 6);
    expr *log_node = new_log(x);
    expr *sum_node = new_sum(log_node, -1);
    sum_node->forward(sum_node, u_vals);
    sum_node->jacobian_init(sum_node);
    sum_node->eval_jacobian(sum_node);
    double expected_Ax[3] = {1.0, 0.5, 1.0 / 3.0};
    int expected_Ap[2] = {0, 3};
    int expected_Ai[3] = {2, 3, 4};

    mu_assert("vals fail", cmp_double_array(sum_node->jacobian->x, expected_Ax, 3));
    mu_assert("rows fail", cmp_int_array(sum_node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(sum_node->jacobian->i, expected_Ai, 3));

    free_expr(sum_node);
    return 0;
}

const char *test_jacobian_sum_mult()
{
    /* Test Jacobian of sum(x * y) where both x and y are 3x1 variables
     * x has global variable index 2, y has global variable index 6
     * Total 10 variables
     * x = [1, 2, 3], y = [2, 3, 4]
     * Jacobian for sum(x*y):
     * d(sum)/dx = [y[0], y[1], y[2]] = [2, 3, 4]
     * d(sum)/dy = [x[0], x[1], x[2]] = [1, 2, 3]
     * Full jacobian with 10 vars: [0, 0, 1, 2, 3, 0, 2, 3, 4, 0]
     */
    double u_vals[10] = {0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0, 4.0, 0.0};

    expr *x = new_variable(3, 1, 2, 10);
    expr *y = new_variable(3, 1, 6, 10);
    expr *mult_node = new_elementwise_mult(x, y);
    expr *sum_node = new_sum(mult_node, -1);

    sum_node->forward(sum_node, u_vals);
    sum_node->jacobian_init(sum_node);
    sum_node->eval_jacobian(sum_node);

    double expected_Ax[6] = {2, 3, 4, 1, 2, 3};
    int expected_Ap[2] = {0, 6}; /* 1x10 matrix: row 0 spans all 6 nonzeros */
    int expected_Ai[6] = {2, 3, 4, 6, 7, 8}; /* column indices */

    mu_assert("vals fail", cmp_double_array(sum_node->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(sum_node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(sum_node->jacobian->i, expected_Ai, 6));

    free_expr(sum_node);
    return 0;
}

const char *test_jacobian_sum_log_axis_0()
{
    /* Test Jacobian of sum(log(x), axis=0) where x is 3x2 variable,
     * global index 2, total 8 variables
     * x.value = [[1.0, 2.0],
     *            [3.0, 4.0],
     *            [5.0, 6.0]]
     * Stored column-wise: [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
     *
     * log(x) = [[log(1), log(2)],
     *           [log(3), log(4)],
     *           [log(5), log(6)]]
     *
     * sum(log(x), axis=0) sums along rows, giving 1x2 result:
     * [log(1) + log(3) + log(5), log(2) + log(4) + log(6)]
     *
     * Jacobian (2 x 8 sparse matrix):
     * Row 0: d(sum[0])/dx = [1/1, 1/3, 1/5] at columns [2, 3, 4]
     * Row 1: d(sum[1])/dx = [1/2, 1/4, 1/6] at columns [5, 6, 7]
     */
    double u_vals[8] = {0.0, 0.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    expr *x = new_variable(3, 2, 2, 8);
    expr *log_node = new_log(x);
    expr *sum_node = new_sum(log_node, 0);
    sum_node->forward(sum_node, u_vals);
    sum_node->jacobian_init(sum_node);
    sum_node->eval_jacobian(sum_node);

    double expected_Ax[6] = {1.0, 1.0 / 3.0, 1.0 / 5.0, 0.5, 0.25, 1.0 / 6.0};
    int expected_Ap[3] = {0, 3, 6};
    int expected_Ai[6] = {2, 3, 4, 5, 6, 7}; /* column indices */

    mu_assert("vals fail", cmp_double_array(sum_node->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(sum_node->jacobian->p, expected_Ap, 3));
    mu_assert("cols fail", cmp_int_array(sum_node->jacobian->i, expected_Ai, 6));

    free_expr(sum_node);
    return 0;
}
const char *test_jacobian_sum_add_log_axis_0()
{
    /* Test Jacobian of sum(add(log(x), log(y)), axis=0) where x and y are 3x2
     * x.value = [[1.0, 2.0],
     *            [3.0, 4.0],
     *            [5.0, 6.0]]
     * y.value = [[1.0, 2.0],
     *            [3.0, 4.0],
     *            [5.0, 6.0]]
     * Stored column-wise: x = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
     *                     y = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
     * Jacobian (2 x 14 sparse matrix):
     * Row 0: d(sum[0])/dx = [1/1, 1/3, 1/5] at columns [2, 3, 4]
     *        d(sum[0])/dy = [1/1, 1/3, 1/5] at columns [8, 9, 10]
     * Row 1: d(sum[1])/dx = [1/2, 1/4, 1/6] at columns [5, 6, 7]
     *        d(sum[1])/dy = [1/2, 1/4, 1/6] at columns [11, 12, 13]
     */
    double u_vals[14] = {0.0, 0.0, 1.0, 3.0, 5.0, 2.0, 4.0,
                         6.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0};

    expr *x = new_variable(3, 2, 2, 14);
    expr *y = new_variable(3, 2, 8, 14);
    expr *log_x = new_log(x);
    expr *log_y = new_log(y);
    expr *add_node = new_add(log_x, log_y);
    expr *sum_node = new_sum(add_node, 0);

    sum_node->forward(sum_node, u_vals);
    sum_node->jacobian_init(sum_node);
    sum_node->eval_jacobian(sum_node);

    /* Expected jacobian values for both rows */
    double expected_Ax[12] = {1.0, 1.0 / 3.0, 1.0 / 5.0,  /* d(sum[0])/dx */
                              1.0, 1.0 / 3.0, 1.0 / 5.0,  /* d(sum[0])/dy */
                              0.5, 0.25,      1.0 / 6.0,  /* d(sum[1])/dx */
                              0.5, 0.25,      1.0 / 6.0}; /* d(sum[1])/dy */
    int expected_Ap[3] = {0, 6, 12};
    int expected_Ai[12] = {2, 3, 4, 8,  9,  10,  /* row 0 columns */
                           5, 6, 7, 11, 12, 13}; /* row 1 columns */

    mu_assert("vals fail", cmp_double_array(sum_node->jacobian->x, expected_Ax, 12));
    mu_assert("rows fail", cmp_int_array(sum_node->jacobian->p, expected_Ap, 3));
    mu_assert("cols fail", cmp_int_array(sum_node->jacobian->i, expected_Ai, 12));

    free_expr(sum_node);
    return 0;
}
const char *test_jacobian_sum_log_axis_1()
{
    /* Test Jacobian of sum(log(x), axis=1) where x is 3x2 variable,
     * global index 2, total 8 variables
     * x.value = [[1.0, 2.0],
     *            [3.0, 4.0],
     *            [5.0, 6.0]]
     * Stored column-wise: [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
     *
     * log(x) = [[log(1), log(2)],
     *           [log(3), log(4)],
     *           [log(5), log(6)]]
     *
     * sum(log(x), axis=1) sums along columns, giving 3x1 result:
     * [log(1) + log(2),
     *  log(3) + log(4),
     *  log(5) + log(6)]
     *
     * Jacobian (3 x 8 sparse matrix):
     * Row 0: d(sum[0])/dx = [1/1, 1/2] at columns [2, 5]
     * Row 1: d(sum[1])/dx = [1/3, 1/4] at columns [3, 6]
     * Row 2: d(sum[2])/dx = [1/5, 1/6] at columns [4, 7]
     */
    double u_vals[8] = {0.0, 0.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    expr *x = new_variable(3, 2, 2, 8);
    expr *log_node = new_log(x);
    expr *sum_node = new_sum(log_node, 1);
    sum_node->forward(sum_node, u_vals);
    sum_node->jacobian_init(sum_node);
    sum_node->eval_jacobian(sum_node);

    double expected_Ax[6] = {1.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 5.0, 1.0 / 6.0};
    int expected_Ap[4] = {0, 2, 4, 6};
    int expected_Ai[6] = {2, 5, 3, 6, 4, 7}; /* column indices */

    mu_assert("vals fail", cmp_double_array(sum_node->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(sum_node->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(sum_node->jacobian->i, expected_Ai, 6));

    free_expr(sum_node);
    return 0;
}

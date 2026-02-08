#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_sum_axis_neg1()
{
    /* Create a 3x2 constant matrix stored column-wise:
       [1, 4]
       [2, 5]
       [3, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_constant(3, 2, 0, values);
    expr *log_node = new_log(const_node);
    expr *sum_node = new_sum(log_node, -1);
    sum_node->forward(sum_node, NULL);

    /* Expected: sum of log(1) + log(2) + log(3) + log(4) + log(5) + log(6) */
    double expected =
        log(1.0) + log(2.0) + log(3.0) + log(4.0) + log(5.0) + log(6.0);

    mu_assert("Sum with axis=-1 test failed",
              fabs(sum_node->value[0] - expected) < 1e-10);

    free_expr(sum_node);
    return 0;
}

const char *test_sum_axis_0()
{
    /* Create a 3x2 constant matrix stored column-wise:
       [1, 4]
       [2, 5]
       [3, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_constant(3, 2, 0, values);
    expr *log_node = new_log(const_node);
    expr *sum_node = new_sum(log_node, 0);
    sum_node->forward(sum_node, NULL);

    /* Expected: sum along rows (axis=0), result is 1x2
       [log(1) + log(2) + log(3), log(4) + log(5) + log(6)]
    */
    double expected[2] = {log(1.0) + log(2.0) + log(3.0),
                          log(4.0) + log(5.0) + log(6.0)};

    mu_assert("Sum with axis=0 test failed",
              cmp_double_array(sum_node->value, expected, 2));

    free_expr(sum_node);
    return 0;
}

const char *test_sum_axis_1()
{
    /* Create a 3x2 constant matrix stored column-wise:
       [1, 4]
       [2, 5]
       [3, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_constant(3, 2, 0, values);
    expr *log_node = new_log(const_node);
    expr *sum_node = new_sum(log_node, 1);
    sum_node->forward(sum_node, NULL);

    /* Expected: sum along columns (axis=1), result is 3x1
       [log(1) + log(4)]
       [log(2) + log(5)]
       [log(3) + log(6)]
    */
    double expected[3] = {log(1.0) + log(4.0), log(2.0) + log(5.0),
                          log(3.0) + log(6.0)};

    mu_assert("Sum with axis=1 test failed",
              cmp_double_array(sum_node->value, expected, 3));

    free_expr(sum_node);
    return 0;
}

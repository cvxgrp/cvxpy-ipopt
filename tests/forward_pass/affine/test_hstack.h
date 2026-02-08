#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_hstack_forward_vectors()
{
    /* x is 3x1 variable with values [1, 2, 3] */
    double u[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Build child nodes */
    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);
    expr *sin_x = new_sin(x);

    expr *args[3] = {log_x, exp_x, sin_x};
    expr *stack = new_hstack(args, 3, 3);

    stack->forward(stack, u);

    double expected[9] = {log(1.0), log(2.0), log(3.0), exp(1.0), exp(2.0),
                          exp(3.0), sin(1.0), sin(2.0), sin(3.0)};

    mu_assert("hstack forward failed", cmp_double_array(stack->value, expected, 9));

    free_expr(stack);
    return 0;
}

const char *test_hstack_forward_matrix()
{
    /* x is 3x2 variable with values stored column-wise: [1, 3, 5, 2, 4, 6] */
    double u[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    expr *x = new_variable(3, 2, 0, 6);

    /* Build child nodes */
    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);
    expr *sin_x = new_sin(x);

    expr *args[3] = {log_x, exp_x, sin_x};
    expr *stack = new_hstack(args, 3, 6);

    stack->forward(stack, u);

    /* vec(log(x)), vec(exp(x)), vec(sin(x)) concatenated column-wise */
    double expected[18] = {log(1.0), log(3.0), log(5.0), log(2.0), log(4.0),
                           log(6.0), exp(1.0), exp(3.0), exp(5.0), exp(2.0),
                           exp(4.0), exp(6.0), sin(1.0), sin(3.0), sin(5.0),
                           sin(2.0), sin(4.0), sin(6.0)};

    mu_assert("hstack forward (3x2) failed",
              cmp_double_array(stack->value, expected, 18));

    free_expr(stack);
    return 0;
}

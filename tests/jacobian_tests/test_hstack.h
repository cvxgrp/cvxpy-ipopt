#include <math.h>
#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_hstack_vectors()
{
    /* Test Jacobian of hstack(log(x), exp(x), sin(x)) where x is 3x1
     * x = [1, 2, 3] at global indices [0, 1, 2]
     * Output is 3x3 matrix (9 elements) stored column-wise:
     * [log(1), log(2), log(3), exp(1), exp(2), exp(3), sin(1), sin(2), sin(3)]
     *
     * Jacobian is 9x3 sparse matrix:
     * d(log(x))/dx: diagonal with [1/1, 1/2, 1/3]
     * d(exp(x))/dx: diagonal with [exp(1), exp(2), exp(3)]
     * d(sin(x))/dx: diagonal with [cos(1), cos(2), cos(3)]
     */
    double u[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);
    expr *sin_x = new_sin(x);

    expr *args[3] = {log_x, exp_x, sin_x};
    expr *stack = new_hstack(args, 3, 3);

    stack->forward(stack, u);
    stack->jacobian_init(stack);
    stack->eval_jacobian(stack);

    /* Expected jacobian: 9x3 with 9 nonzeros (diagonal blocks) */
    double expected_Ax[9] = {1.0,      0.5,      1.0 / 3.0, /* d(log)/dx */
                             exp(1.0), exp(2.0), exp(3.0),  /* d(exp)/dx */
                             cos(1.0), cos(2.0), cos(3.0)}; /* d(sin)/dx */

    int expected_Ai[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int expected_Ap[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    mu_assert("vals fail", cmp_double_array(stack->jacobian->x, expected_Ax, 9));
    mu_assert("cols fail", cmp_int_array(stack->jacobian->i, expected_Ai, 9));
    mu_assert("rows fail", cmp_int_array(stack->jacobian->p, expected_Ap, 10));

    free_expr(stack);
    return 0;
}

const char *test_jacobian_hstack_matrix()
{
    /* Test Jacobian of hstack(log(x), exp(x), sin(x)) where x is 3x2
     * x stored column-wise: [1, 3, 5, 2, 4, 6]
     * Output is 3x6 matrix (18 elements):
     * [log(1), log(3), log(5), log(2), log(4), log(6),
     *  exp(1), exp(3), exp(5), exp(2), exp(4), exp(6),
     *  sin(1), sin(3), sin(5), sin(2), sin(4), sin(6)]
     *
     * Jacobian is 18x6 sparse matrix (diagonal):
     * Each child contributes a 6x6 diagonal block
     */
    double u[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    expr *x = new_variable(3, 2, 0, 6);

    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);
    expr *sin_x = new_sin(x);

    expr *args[3] = {log_x, exp_x, sin_x};
    expr *stack = new_hstack(args, 3, 6);

    stack->forward(stack, u);
    stack->jacobian_init(stack);
    stack->eval_jacobian(stack);

    /* Expected jacobian: 18x6 with 18 nonzeros (diagonal) */
    double expected_Ax[18] = {
        1.0,      1.0 / 3.0, 1.0 / 5.0, 0.5,      0.25,     1.0 / 6.0, /* log */
        exp(1.0), exp(3.0),  exp(5.0),  exp(2.0), exp(4.0), exp(6.0),  /* exp */
        cos(1.0), cos(3.0),  cos(5.0),  cos(2.0), cos(4.0), cos(6.0)}; /* sin */

    int expected_Ai[18] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    int expected_Ap[19] = {0,  1,  2,  3,  4,  5,  6,  7,  8, 9,
                           10, 11, 12, 13, 14, 15, 16, 17, 18};

    mu_assert("vals fail", cmp_double_array(stack->jacobian->x, expected_Ax, 18));
    mu_assert("cols fail", cmp_int_array(stack->jacobian->i, expected_Ai, 18));
    mu_assert("rows fail", cmp_int_array(stack->jacobian->p, expected_Ap, 19));

    free_expr(stack);
    return 0;
}

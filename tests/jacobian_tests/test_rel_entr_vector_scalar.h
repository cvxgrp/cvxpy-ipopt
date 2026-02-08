#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_jacobian_rel_entr_vector_scalar()
{
    // x is 3x1 with values [1, 2, 3], y is scalar 4
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    expr *x = new_variable(3, 1, 0, 4);
    expr *y = new_variable(1, 1, 3, 4);
    expr *node = new_rel_entr_second_arg_scalar(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double a = log(1.0 / 4.0) + 1.0;
    double b = log(2.0 / 4.0) + 1.0;
    double c = log(3.0 / 4.0) + 1.0;
    double d = -1.0 / 4.0;
    double e = -2.0 / 4.0;
    double f = -3.0 / 4.0;

    double expected_Ax[6] = {a, d, b, e, c, f};
    int expected_Ap[4] = {0, 2, 4, 6};
    int expected_Ai[6] = {0, 3, 1, 3, 2, 3};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 6));
    free_expr(node);
    return 0;
}

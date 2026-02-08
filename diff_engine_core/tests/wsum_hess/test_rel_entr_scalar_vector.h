#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_rel_entr_scalar_vector()
{
    // x is scalar 1, y is 3x1 with values [2, 3, 4], w = [4, 5, 6]
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[3] = {4.0, 5.0, 6.0};

    expr *x = new_variable(1, 1, 0, 4);
    expr *y = new_variable(3, 1, 1, 4);
    expr *node = new_rel_entr_first_arg_scalar(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, w);

    int expected_p[5] = {0, 4, 6, 8, 10};
    int expected_i[10] = {0, 1, 2, 3, 0, 1, 0, 2, 0, 3};
    double expected_x[10] = {15.0, -2.0,       -5.0 / 3.0, -1.5, -2.0,
                             1.0,  -5.0 / 3.0, 5.0 / 9.0,  -1.5, 0.375};

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 5));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 10));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 10));

    free_expr(node);
    return 0;
}

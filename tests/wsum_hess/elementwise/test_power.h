#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_power()
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *power_node = new_power(x, 3.0);
    power_node->forward(power_node, u_vals);
    power_node->wsum_hess_init(power_node);
    power_node->eval_wsum_hess(power_node, w);

    /* Expected values on the diagonal: w_i * 6 * x_i */
    double expected_x[3] = {6.0 * 1.0, 6.0 * 4.0, 6.0 * 9.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(power_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(power_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(power_node->wsum_hess->i, expected_i, 3));

    free_expr(power_node);

    return 0;
}

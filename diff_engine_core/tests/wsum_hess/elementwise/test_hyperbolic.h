#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_sinh()
{
    /* Test: wsum_hess of sinh(x) where x = [1, 2, 3] (3x1) at global variable index
     * 0 Total 3 variables, weight w = [1, 2, 3]
     *
     * For sinh(x), the Hessian is sinh(x)
     * Weighted sum of Hessian diagonal: w_i * sinh(x_i)
     */

    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *sinh_node = new_sinh(x);
    sinh_node->forward(sinh_node, u_vals);
    sinh_node->wsum_hess_init(sinh_node);
    sinh_node->eval_wsum_hess(sinh_node, w);

    /* Expected values on the diagonal: w_i * sinh(x_i) */
    double expected_x[3] = {1.0 * sinh(1.0), 2.0 * sinh(2.0), 3.0 * sinh(3.0)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(sinh_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(sinh_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(sinh_node->wsum_hess->i, expected_i, 3));

    free_expr(sinh_node);

    return 0;
}

const char *test_wsum_hess_tanh()
{
    /* Test: wsum_hess of tanh(x) where x = [1, 2, 3] (3x1) at global variable index
     * 0 Total 3 variables, weight w = [1, 2, 3]
     *
     * For tanh(x), the Hessian is -2*tanh(x)/cosh^2(x)
     * Weighted sum of Hessian diagonal: w_i * (-2*tanh(x_i)/cosh^2(x_i))
     */

    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *tanh_node = new_tanh(x);
    tanh_node->forward(tanh_node, u_vals);
    tanh_node->wsum_hess_init(tanh_node);
    tanh_node->eval_wsum_hess(tanh_node, w);

    /* Expected values on the diagonal: w_i * (-2*tanh(x_i)/cosh^2(x_i)) */
    double expected_x[3] = {1.0 * (-2.0 * tanh(1.0) / pow(cosh(1.0), 2)),
                            2.0 * (-2.0 * tanh(2.0) / pow(cosh(2.0), 2)),
                            3.0 * (-2.0 * tanh(3.0) / pow(cosh(3.0), 2))};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(tanh_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(tanh_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(tanh_node->wsum_hess->i, expected_i, 3));

    free_expr(tanh_node);

    return 0;
}

const char *test_wsum_hess_asinh()
{
    /* Test: wsum_hess of asinh(x) where x = [1, 2, 3] (3x1) at global variable index
     * 0 Total 3 variables, weight w = [1, 2, 3]
     *
     * For asinh(x), the Hessian is -x/(1+x^2)^(3/2)
     * Weighted sum of Hessian diagonal: w_i * (-x_i/(1+x_i^2)^(3/2))
     */

    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *asinh_node = new_asinh(x);
    asinh_node->forward(asinh_node, u_vals);
    asinh_node->wsum_hess_init(asinh_node);
    asinh_node->eval_wsum_hess(asinh_node, w);

    /* Expected values on the diagonal: w_i * (-x_i/(1+x_i^2)^(3/2)) */
    double expected_x[3] = {1.0 * (-1.0 / pow(1.0 + 1.0 * 1.0, 1.5)),
                            2.0 * (-2.0 / pow(1.0 + 2.0 * 2.0, 1.5)),
                            3.0 * (-3.0 / pow(1.0 + 3.0 * 3.0, 1.5))};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(asinh_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(asinh_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(asinh_node->wsum_hess->i, expected_i, 3));

    free_expr(asinh_node);

    return 0;
}

const char *test_wsum_hess_atanh()
{
    /* Test: wsum_hess of atanh(x) where x = [0.1, 0.2, 0.3] (3x1) at global variable
     * index 0 Total 3 variables, weight w = [1, 2, 3]
     *
     * For atanh(x), the Hessian is 2*x/(1-x^2)^2
     * Weighted sum of Hessian diagonal: w_i * (2*x_i/(1-x_i^2)^2)
     * Note: using smaller values for x to avoid domain issues (|x| < 1)
     */

    double u_vals[3] = {0.1, 0.2, 0.3};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *atanh_node = new_atanh(x);
    atanh_node->forward(atanh_node, u_vals);
    atanh_node->wsum_hess_init(atanh_node);
    atanh_node->eval_wsum_hess(atanh_node, w);

    /* Expected values on the diagonal: w_i * (2*x_i/(1-x_i^2)^2) */
    double expected_x[3] = {1.0 * (2.0 * 0.1 / pow(1.0 - 0.1 * 0.1, 2)),
                            2.0 * (2.0 * 0.2 / pow(1.0 - 0.2 * 0.2, 2)),
                            3.0 * (2.0 * 0.3 / pow(1.0 - 0.3 * 0.3, 2))};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(atanh_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(atanh_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(atanh_node->wsum_hess->i, expected_i, 3));

    free_expr(atanh_node);

    return 0;
}

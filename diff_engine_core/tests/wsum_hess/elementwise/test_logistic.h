#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_logistic()
{
    /* Test: wsum_hess of logistic(x) where x = [1, 2, 3] (3x1) at global variable
     * index 0 Total 3 variables, weight w = [1, 2, 3]
     *
     * For logistic(x) = log(1+exp(x)), the Hessian is σ(x)*(1-σ(x))
     * where σ(x) = exp(x)/(1+exp(x)) = 1/(1+exp(-x))
     * Weighted sum of Hessian diagonal: w_i * σ(x_i) * (1 - σ(x_i))
     */

    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *logistic_node = new_logistic(x);
    logistic_node->forward(logistic_node, u_vals);
    logistic_node->jacobian_init(logistic_node);
    logistic_node->eval_jacobian(logistic_node);
    logistic_node->wsum_hess_init(logistic_node);
    logistic_node->eval_wsum_hess(logistic_node, w);

    /* Expected values on the diagonal: w_i * σ(x_i) * (1 - σ(x_i)) */
    double sigma1 = 1.0 / (1.0 + exp(-1.0));
    double sigma2 = 1.0 / (1.0 + exp(-2.0));
    double sigma3 = 1.0 / (1.0 + exp(-3.0));
    double expected_x[3] = {1.0 * sigma1 * (1.0 - sigma1),
                            2.0 * sigma2 * (1.0 - sigma2),
                            3.0 * sigma3 * (1.0 - sigma3)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(logistic_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(logistic_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(logistic_node->wsum_hess->i, expected_i, 3));

    free_expr(logistic_node);

    return 0;
}

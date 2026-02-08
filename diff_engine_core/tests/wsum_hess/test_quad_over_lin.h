#include "affine.h"
#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_quad_over_lin_xy()
{
    /* x^T x / y with x var_id=2 (3 vars), y var_id=7, total n_vars=9
     * x = [1, 2, 3], y = 4, w = 2
     */
    double u_vals[9] = {0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0};
    double w = 2.0;

    expr *x = new_variable(3, 1, 2, 9);
    expr *y = new_variable(1, 1, 7, 9);
    expr *node = new_quad_over_lin(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, &w);

    int expected_p[10] = {0, 0, 0, 2, 4, 6, 6, 6, 10, 10};
    int expected_i[10] = {2, 7, 3, 7, 4, 7, 2, 3, 4, 7};
    double expected_x[10] = {1.0,   -0.25, 1.0,  -0.5,  1.0,
                             -0.75, -0.25, -0.5, -0.75, 0.875};

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 10));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 10));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 10));

    free_expr(node);
    return 0;
}

const char *test_wsum_hess_quad_over_lin_yx()
{
    /* x^T x / y with y var_id=2, x var_id=5 (3 vars), total n_vars=9
     * x = [1, 2, 3], y = 4, w = 2
     */
    double u_vals[9] = {0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0};
    double w = 2.0;

    expr *y = new_variable(1, 1, 2, 9);
    expr *x = new_variable(3, 1, 5, 9);
    expr *node = new_quad_over_lin(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, &w);

    int expected_p[10] = {0, 0, 0, 4, 4, 4, 6, 8, 10, 10};
    int expected_i[10] = {2, 5, 6, 7, 2, 5, 2, 6, 2, 7};
    double expected_x[10] = {0.875, -0.25, -0.5, -0.75, -0.25,
                             1.0,   -0.5,  1.0,  -0.75, 1.0};

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 10));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 10));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 10));

    free_expr(node);
    return 0;
}

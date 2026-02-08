// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_index_log(void)
{
    /* log(x)[0, 2] with w = [1, 2]
     * Hessian of log is diag(-1/x^2)
     * After scatter: parent_w = [1, 0, 2]
     * Hessian: diag(-1/x^2) weighted by [1, 0, 2] = diag([-1, 0, -0.125]) */
    double u[3] = {1.0, 2.0, 4.0};
    int indices[2] = {0, 2};
    double w[2] = {1.0, 2.0};

    expr *var = new_variable(3, 1, 0, 3);
    expr *log_node = new_log(var);
    expr *idx = new_index(log_node, 1, 2, indices, 2);

    idx->forward(idx, u);
    idx->jacobian_init(idx);
    idx->wsum_hess_init(idx);
    idx->eval_wsum_hess(idx, w);

    /* Expected diagonal values:
     * H[0,0] = -1 * 1/1^2 = -1.0
     * H[1,1] = 0 (no weight)
     * H[2,2] = -2 * 1/16 = -0.125 */
    double expected_x[3] = {-1.0, 0.0, -0.125};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("index log hess vals",
              cmp_double_array(idx->wsum_hess->x, expected_x, 3));
    mu_assert("index log hess p", cmp_int_array(idx->wsum_hess->p, expected_p, 4));
    mu_assert("index log hess i", cmp_int_array(idx->wsum_hess->i, expected_i, 3));

    free_expr(idx);
    return 0;
}

const char *test_wsum_hess_index_repeated(void)
{
    /* log(x)[0, 0] with w = [1, 2] - weights should accumulate
     * parent_w = [3, 0, 0] (1+2 at position 0) */
    double u[3] = {2.0, 3.0, 4.0};
    int indices[2] = {0, 0};
    double w[2] = {1.0, 2.0};

    expr *var = new_variable(3, 1, 0, 3);
    expr *log_node = new_log(var);
    expr *idx = new_index(log_node, 1, 2, indices, 2);

    idx->forward(idx, u);
    idx->jacobian_init(idx);
    idx->wsum_hess_init(idx);
    idx->eval_wsum_hess(idx, w);

    /* Hessian of log at x=2 is -1/4
     * weighted by 3 (accumulated) -> -3/4 = -0.75
     * Other positions have 0 weight */
    double expected_x[3] = {-0.75, 0.0, 0.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("index repeated hess vals",
              cmp_double_array(idx->wsum_hess->x, expected_x, 3));
    mu_assert("index repeated hess p",
              cmp_int_array(idx->wsum_hess->p, expected_p, 4));
    mu_assert("index repeated hess i",
              cmp_int_array(idx->wsum_hess->i, expected_i, 3));

    free_expr(idx);
    return 0;
}

const char *test_wsum_hess_sum_index_log(void)
{
    /* sum(log(x)[0, 2]) with w = 1.0
     * This tests the chain: sum -> index -> log -> variable
     * Gradient: [1/x[0], 0, 1/x[2]]
     * Hessian: diag([-1/x[0]^2, 0, -1/x[2]^2]) */
    double u[3] = {1.0, 2.0, 4.0};
    int indices[2] = {0, 2};
    double w = 1.0;

    expr *var = new_variable(3, 1, 0, 3);
    expr *log_node = new_log(var);
    expr *idx = new_index(log_node, 1, 2, indices, 2);
    expr *sum_node = new_sum(idx, -1);

    sum_node->forward(sum_node, u);
    sum_node->jacobian_init(sum_node);
    sum_node->wsum_hess_init(sum_node);
    sum_node->eval_wsum_hess(sum_node, &w);

    /* Expected diagonal values:
     * H[0,0] = -1 * 1/1^2 = -1.0
     * H[1,1] = 0 (index 1 not selected)
     * H[2,2] = -1 * 1/16 = -0.0625 */
    double expected_x[3] = {-1.0, 0.0, -0.0625};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("sum index log hess vals",
              cmp_double_array(sum_node->wsum_hess->x, expected_x, 3));
    mu_assert("sum index log hess p",
              cmp_int_array(sum_node->wsum_hess->p, expected_p, 4));
    mu_assert("sum index log hess i",
              cmp_int_array(sum_node->wsum_hess->i, expected_i, 3));

    free_expr(sum_node);
    return 0;
}

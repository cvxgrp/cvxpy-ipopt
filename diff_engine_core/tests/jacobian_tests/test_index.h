// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_index_forward_simple(void)
{
    /* x = [1, 2, 3], select indices [0, 2] -> [1, 3] */
    double u[3] = {1.0, 2.0, 3.0};
    int indices[2] = {0, 2};
    expr *var = new_variable(3, 1, 0, 3);
    expr *idx = new_index(var, 1, 2, indices, 2);
    idx->forward(idx, u);

    double expected[2] = {1.0, 3.0};
    mu_assert("index forward simple", cmp_double_array(idx->value, expected, 2));

    free_expr(idx);
    return 0;
}

const char *test_index_forward_repeated(void)
{
    /* x = [1, 2, 3], select [0, 0, 2] -> [1, 1, 3] */
    double u[3] = {1.0, 2.0, 3.0};
    int indices[3] = {0, 0, 2};
    expr *var = new_variable(3, 1, 0, 3);
    expr *idx = new_index(var, 1, 3, indices, 3);
    idx->forward(idx, u);

    double expected[3] = {1.0, 1.0, 3.0};
    mu_assert("index forward repeated", cmp_double_array(idx->value, expected, 3));

    free_expr(idx);
    return 0;
}

const char *test_index_jacobian_of_variable(void)
{
    /* x[0, 2] where x is variable of size 3
     * Jacobian should select rows 0 and 2 of identity */
    double u[3] = {1.0, 2.0, 3.0};
    int indices[2] = {0, 2};
    expr *var = new_variable(3, 1, 0, 3);
    expr *idx = new_index(var, 1, 2, indices, 2);
    idx->forward(idx, u);
    idx->jacobian_init(idx);
    idx->eval_jacobian(idx);

    /* Jacobian is 2x3 with pattern: row 0 selects col 0, row 1 selects col 2 */
    double expected_x[2] = {1.0, 1.0};
    int expected_p[3] = {0, 1, 2}; /* CSR row ptrs */
    int expected_i[2] = {0, 2};    /* column indices */

    mu_assert("index jac vals", cmp_double_array(idx->jacobian->x, expected_x, 2));
    mu_assert("index jac p", cmp_int_array(idx->jacobian->p, expected_p, 3));
    mu_assert("index jac i", cmp_int_array(idx->jacobian->i, expected_i, 2));

    free_expr(idx);
    return 0;
}

const char *test_index_jacobian_of_log(void)
{
    /* log(x)[0, 2] - jacobian should be selected rows of diag(1/x) */
    double u[3] = {1.0, 2.0, 4.0};
    int indices[2] = {0, 2};
    expr *var = new_variable(3, 1, 0, 3);
    expr *log_node = new_log(var);
    expr *idx = new_index(log_node, 1, 2, indices, 2);
    idx->forward(idx, u);
    idx->jacobian_init(idx);
    idx->eval_jacobian(idx);

    /* d/dx log(x) = diag(1/x), then select rows 0 and 2
     * Row 0: 1/1 = 1.0 at col 0
     * Row 1: 1/4 = 0.25 at col 2 */
    double expected_x[2] = {1.0, 0.25};
    int expected_i[2] = {0, 2};

    mu_assert("index of log jac vals",
              cmp_double_array(idx->jacobian->x, expected_x, 2));
    mu_assert("index of log jac cols",
              cmp_int_array(idx->jacobian->i, expected_i, 2));

    free_expr(idx);
    return 0;
}

const char *test_index_jacobian_repeated(void)
{
    /* x[0, 0] where x is size 3 - both outputs depend on x[0] */
    double u[3] = {1.0, 2.0, 3.0};
    int indices[2] = {0, 0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *idx = new_index(var, 1, 2, indices, 2);
    idx->forward(idx, u);
    idx->jacobian_init(idx);
    idx->eval_jacobian(idx);

    /* Both rows select column 0 */
    double expected_x[2] = {1.0, 1.0};
    int expected_p[3] = {0, 1, 2};
    int expected_i[2] = {0, 0}; /* Both reference col 0 */

    mu_assert("index repeated jac vals",
              cmp_double_array(idx->jacobian->x, expected_x, 2));
    mu_assert("index repeated row ptr",
              cmp_int_array(idx->jacobian->p, expected_p, 3));
    mu_assert("index repeated jac i",
              cmp_int_array(idx->jacobian->i, expected_i, 2));

    free_expr(idx);
    return 0;
}

const char *test_sum_of_index(void)
{
    /* sum(x[0, 2]) = x[0] + x[2]
     * Gradient should be sparse with entries at cols 0 and 2 */
    double u[3] = {1.0, 2.0, 3.0};
    int indices[2] = {0, 2};

    expr *var = new_variable(3, 1, 0, 3);
    expr *idx = new_index(var, 1, 2, indices, 2);
    expr *s = new_sum(idx, -1); /* sum all */

    s->forward(s, u);
    s->jacobian_init(s);
    s->eval_jacobian(s);

    /* Gradient: [1, 0, 1] in sparse form */
    double expected_x[2] = {1.0, 1.0};
    int expected_i[2] = {0, 2};

    mu_assert("sum of index vals", cmp_double_array(s->jacobian->x, expected_x, 2));
    mu_assert("sum of index cols", cmp_int_array(s->jacobian->i, expected_i, 2));

    free_expr(s);
    return 0;
}

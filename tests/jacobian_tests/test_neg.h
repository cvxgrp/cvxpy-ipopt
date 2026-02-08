#include <stdio.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_neg_jacobian(void)
{
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *neg_node = new_neg(var);
    neg_node->forward(neg_node, u);
    neg_node->jacobian_init(neg_node);
    neg_node->eval_jacobian(neg_node);

    /* Jacobian of neg(x) is -I (diagonal with -1) */
    double expected_x[3] = {-1.0, -1.0, -1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("neg jacobian vals fail",
              cmp_double_array(neg_node->jacobian->x, expected_x, 3));
    mu_assert("neg jacobian rows fail",
              cmp_int_array(neg_node->jacobian->p, expected_p, 4));
    mu_assert("neg jacobian cols fail",
              cmp_int_array(neg_node->jacobian->i, expected_i, 3));

    free_expr(neg_node);
    return 0;
}

const char *test_neg_chain(void)
{
    /* Test neg(neg(x)) = x */
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *neg1 = new_neg(var);
    expr *neg2 = new_neg(neg1);
    neg2->forward(neg2, u);

    /* neg(neg(x)) should equal x */
    mu_assert("neg chain forward failed", cmp_double_array(neg2->value, u, 3));

    neg2->jacobian_init(neg2);
    neg2->eval_jacobian(neg2);

    /* Jacobian of neg(neg(x)) is (-1)*(-1)*I = I */
    double expected_x[3] = {1.0, 1.0, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("neg chain jacobian vals fail",
              cmp_double_array(neg2->jacobian->x, expected_x, 3));
    mu_assert("neg chain jacobian rows fail",
              cmp_int_array(neg2->jacobian->p, expected_p, 4));
    mu_assert("neg chain jacobian cols fail",
              cmp_int_array(neg2->jacobian->i, expected_i, 3));

    free_expr(neg2);
    return 0;
}

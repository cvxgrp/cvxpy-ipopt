#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_promote_scalar_jacobian(void)
{
    /* Promote scalar to 3-element vector, check jacobian */
    double u[1] = {2.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 3, 1);
    promote_node->forward(promote_node, u);
    promote_node->jacobian_init(promote_node);
    promote_node->eval_jacobian(promote_node);

    /* Jacobian is 3x1 with all 1s (each output depends on same input) */
    double expected_x[3] = {1.0, 1.0, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 0, 0};

    mu_assert("promote jacobian vals fail",
              cmp_double_array(promote_node->jacobian->x, expected_x, 3));
    mu_assert("promote jacobian rows fail",
              cmp_int_array(promote_node->jacobian->p, expected_p, 4));
    mu_assert("promote jacobian cols fail",
              cmp_int_array(promote_node->jacobian->i, expected_i, 3));

    free_expr(promote_node);
    return 0;
}

const char *test_promote_scalar_to_matrix_jacobian(void)
{
    /* Promote scalar to 2x3 matrix, check jacobian */
    double u[1] = {7.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 2, 3);
    promote_node->forward(promote_node, u);

    /* Forward: all 6 elements should be 7.0 */
    double expected_val[6] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    mu_assert("promote scalar->matrix forward failed",
              cmp_double_array(promote_node->value, expected_val, 6));

    promote_node->jacobian_init(promote_node);
    promote_node->eval_jacobian(promote_node);

    /* Jacobian is 6x1 with all 1s (each output depends on same scalar input) */
    double expected_x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 0, 0, 0, 0, 0};

    mu_assert("promote matrix jacobian vals fail",
              cmp_double_array(promote_node->jacobian->x, expected_x, 6));
    mu_assert("promote matrix jacobian rows fail",
              cmp_int_array(promote_node->jacobian->p, expected_p, 7));
    mu_assert("promote matrix jacobian cols fail",
              cmp_int_array(promote_node->jacobian->i, expected_i, 6));

    free_expr(promote_node);
    return 0;
}

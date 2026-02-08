#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_promote_scalar_to_vector(void)
{
    /* Promote scalar to 4-element vector */
    double u[1] = {5.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 4, 1);
    promote_node->forward(promote_node, u);

    double expected[4] = {5.0, 5.0, 5.0, 5.0};
    mu_assert("promote scalar->vector forward failed",
              cmp_double_array(promote_node->value, expected, 4));

    free_expr(promote_node);
    return 0;
}

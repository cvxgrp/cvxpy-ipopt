#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_neg_forward(void)
{
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *neg_node = new_neg(var);
    neg_node->forward(neg_node, u);
    double expected[3] = {-1.0, -2.0, -3.0};
    mu_assert("neg forward failed", cmp_double_array(neg_node->value, expected, 3));
    free_expr(neg_node);
    return 0;
}

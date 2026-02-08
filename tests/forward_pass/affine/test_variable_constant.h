#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_variable()
{
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    var->forward(var, u);
    mu_assert("Variable test failed", cmp_double_array(var->value, u, 3));
    free_expr(var);
    return 0;
}

const char *test_constant()
{
    double c[2] = {5.0, 10.0};
    double u[2] = {0.0, 0.0};
    expr *const_node = new_constant(2, 1, 0, c);
    const_node->forward(const_node, u);
    mu_assert("Constant test failed", cmp_double_array(const_node->value, c, 2));
    free_expr(const_node);
    return 0;
}

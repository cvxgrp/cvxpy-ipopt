#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_addition()
{
    double u[2] = {3.0, 4.0};
    double c[2] = {1.0, 2.0};
    expr *var = new_variable(2, 1, 0, 2);
    expr *const_node = new_constant(2, 1, 0, c);
    expr *sum = new_add(var, const_node);
    sum->forward(sum, u);
    double expected[2] = {4.0, 6.0};
    mu_assert("Addition test failed", cmp_double_array(sum->value, expected, 2));
    free_expr(sum);
    return 0;
}

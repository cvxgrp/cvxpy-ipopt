#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_exp()
{
    double u[2] = {0.0, 1.0};
    expr *var = new_variable(2, 1, 0, 2);
    expr *exp_node = new_exp(var);
    exp_node->forward(exp_node, u);
    double correct[2] = {exp(0.0), exp(1.0)};
    mu_assert("fail", cmp_double_array(exp_node->value, correct, 2));
    free_expr(exp_node);
    return 0;
}

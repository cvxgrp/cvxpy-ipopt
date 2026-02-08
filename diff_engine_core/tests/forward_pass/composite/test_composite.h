#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_composite()
{
    double u[2] = {1.0, 2.0};
    double c[2] = {1.0, 1.0};

    /* Build tree: log(exp(x) + c) */
    expr *var = new_variable(2, 1, 0, 2);
    expr *exp_node = new_exp(var);
    expr *const_node = new_constant(2, 1, 0, c);
    expr *sum = new_add(exp_node, const_node);
    expr *log_node = new_log(sum);

    log_node->forward(log_node, u);

    double correct[2] = {log(exp(1.0) + 1.0), log(exp(2.0) + 1.0)};
    mu_assert("failed", cmp_double_array(log_node->value, correct, 2));

    free_expr(log_node);
    return 0;
}

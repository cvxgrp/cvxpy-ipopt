#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_log()
{
    double u[2] = {1.0, 2.718281828};
    expr *var = new_variable(2, 1, 0, 2);
    expr *log_node = new_log(var);
    log_node->forward(log_node, u);
    double correct[2] = {log(1.0), log(2.718281828)};
    mu_assert("fail", cmp_double_array(log_node->value, correct, 2));
    free_expr(log_node);
    return 0;
}

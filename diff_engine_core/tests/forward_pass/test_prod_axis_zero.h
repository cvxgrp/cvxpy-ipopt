#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_forward_prod_axis_zero()
{
    /* Create a 2x3 constant matrix stored column-wise:
       [1, 3, 5]
       [2, 4, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_constant(2, 3, 0, values);
    expr *prod_node = new_prod_axis_zero(const_node);
    prod_node->forward(prod_node, NULL);

    /* Expected: columnwise products, result is 1x3
       [1*2, 3*4, 5*6] = [2, 12, 30]
    */
    double expected[3] = {2.0, 12.0, 30.0};

    mu_assert("prod_axis_zero forward test failed",
              cmp_double_array(prod_node->value, expected, 3));

    free_expr(prod_node);
    return 0;
}

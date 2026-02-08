#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_forward_prod_axis_one()
{
    /* Create a 2x3 constant matrix stored column-wise:
       [1, 3, 5]
       [2, 4, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_constant(2, 3, 0, values);
    expr *prod_node = new_prod_axis_one(const_node);
    prod_node->forward(prod_node, NULL);

    /* Expected: rowwise products, result is 1x2 (row vector)
       Row 0: 1*3*5 = 15
       Row 1: 2*4*6 = 48
       Result: [15, 48]
    */
    double expected[2] = {15.0, 48.0};

    mu_assert("prod_axis_one forward test failed",
              cmp_double_array(prod_node->value, expected, 2));

    free_expr(prod_node);
    return 0;
}

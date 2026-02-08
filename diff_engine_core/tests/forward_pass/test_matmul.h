#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_matmul()
{
    /* Test: Z = X @ Y where
     * X is 3x2:  [1  4]     Y is 2x4:  [7   9   11  13]
     *            [2  5]                [8   10  12  14]
     *            [3  6]
     *
     * Expected Z (3x4):  [39   49   59   69]
     *                    [54   68   82   96]
     *                    [69   87  105  123]
     */

    /* Create X variable (3 x 2) - stored in column-major order */
    double u_x[6] = {1.0, 2.0, 3.0,  /* first column */
                     4.0, 5.0, 6.0}; /* second column */
    expr *X = new_variable(3, 2, 0, 20);

    /* Create Y variable (2 x 4) - stored in column-major order */
    double u_y[8] = {7.0,  8.0,   /* first column */
                     9.0,  10.0,  /* second column */
                     11.0, 12.0,  /* third column */
                     13.0, 14.0}; /* fourth column */
    expr *Y = new_variable(2, 4, 6, 20);

    /* Create matmul expression */
    expr *Z = new_matmul(X, Y);

    /* Concatenate parameter vectors */
    double u[14];
    for (int i = 0; i < 6; i++) u[i] = u_x[i];
    for (int i = 0; i < 8; i++) u[6 + i] = u_y[i];

    /* Evaluate forward pass */
    Z->forward(Z, u);

    /* Expected result (3 x 4) in column-major order */
    double expected[12] = {39.0, 54.0, 69.0,   /* first column */
                           49.0, 68.0, 87.0,   /* second column */
                           59.0, 82.0, 105.0,  /* third column */
                           69.0, 96.0, 123.0}; /* fourth column */

    /* Verify dimensions */
    mu_assert("matmul result should have d1=3", Z->d1 == 3);
    mu_assert("matmul result should have d2=4", Z->d2 == 4);
    mu_assert("matmul result should have size=12", Z->size == 12);

    /* Verify values */
    mu_assert("Matmul forward pass test failed",
              cmp_double_array(Z->value, expected, 12));

    free_expr(Z);
    return 0;
}

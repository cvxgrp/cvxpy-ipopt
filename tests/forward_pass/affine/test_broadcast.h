#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_broadcast_row()
{
    /* Test broadcast row: (1, 3) -> (2, 3)
     * Input:  [1.0, 2.0, 3.0] (row vector)
     * Output: [[1.0, 2.0, 3.0],
     *          [1.0, 2.0, 3.0]]
     * Vectorized columnwise: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
     */
    double row_data[3] = {1.0, 2.0, 3.0};
    expr *row_var = new_variable(1, 3, 0, 3);
    expr *bcast = new_broadcast(row_var, 2, 3);

    bcast->forward(bcast, row_data);

    /* Expected: columnwise vectorization [col1, col2, col3] */
    double expected[6] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    mu_assert("Broadcast row test failed",
              cmp_double_array(bcast->value, expected, 6));

    free_expr(bcast);
    return 0;
}

const char *test_broadcast_col()
{
    /* Test broadcast column: (3, 1) -> (3, 2)
     * Input:  [[1.0],
     *          [2.0],
     *          [3.0]] (column vector)
     * Output: [[1.0, 1.0],
     *          [2.0, 2.0],
     *          [3.0, 3.0]]
     * Vectorized columnwise: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
     */
    double col_data[3] = {1.0, 2.0, 3.0};
    expr *col_var = new_variable(3, 1, 0, 3);
    expr *bcast = new_broadcast(col_var, 3, 2);

    bcast->forward(bcast, col_data);

    /* Expected: columnwise vectorization [col1, col2] */
    double expected[6] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    mu_assert("Broadcast column test failed",
              cmp_double_array(bcast->value, expected, 6));

    free_expr(bcast);
    return 0;
}

const char *test_broadcast_matrix()
{
    /* Test no broadcast needed: (2, 3) -> (2, 3)
     * This should work when child shape already matches target
     * Actually, based on the implementation, broadcast is only for:
     * - row: (1, n) -> (m, n)
     * - col: (m, 1) -> (m, n)
     * - scalar: (1, 1) -> (m, n)
     * So let's test scalar broadcast instead.
     */

    /* Test scalar broadcast: (1, 1) -> (2, 3)
     * Input:  [5.0] (scalar)
     * Output: [[5.0, 5.0, 5.0],
     *          [5.0, 5.0, 5.0]]
     * Vectorized columnwise: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
     */
    double scalar_data[1] = {5.0};
    expr *scalar_var = new_variable(1, 1, 0, 1);
    expr *bcast = new_broadcast(scalar_var, 2, 3);

    bcast->forward(bcast, scalar_data);

    /* Expected: all elements are 5.0 */
    double expected[6] = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    mu_assert("Broadcast scalar test failed",
              cmp_double_array(bcast->value, expected, 6));

    free_expr(bcast);
    return 0;
}

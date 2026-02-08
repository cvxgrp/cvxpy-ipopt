#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_broadcast_row_jacobian()
{
    /* Test jacobian of broadcast row: (1, 3) -> (2, 3)
     * Input variable: 3 elements (d1=1, d2=3)
     * Output: 6 elements (d1=2, d2=3) stored columnwise
     *
     * For a row broadcast, each column of the child is repeated m times.
     * Child jacobian (3x1 sparsity):
     *   var[0] -> out[0]
     *   var[1] -> out[1]
     *   var[2] -> out[2]
     *
     * Broadcast row (1,3) -> (2,3) repeats the row 2 times:
     * Broadcast jacobian (6x1 sparsity, columnwise):
     *   var[0] -> out[0], out[1]  (column 0, rows 0,1)
     *   var[1] -> out[2], out[3]  (column 1, rows 0,1)
     *   var[2] -> out[4], out[5]  (column 2, rows 0,1)
     */
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(1, 3, 0, 3);
    expr *bcast = new_broadcast(var, 2, 3);
    bcast->forward(bcast, u);
    bcast->jacobian_init(bcast);
    bcast->eval_jacobian(bcast);

    /* Each variable affects 2 elements (m times) */
    double expected_x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 0, 1, 1, 2, 2};

    mu_assert("broadcast row jacobian vals fail",
              cmp_double_array(bcast->jacobian->x, expected_x, 6));
    mu_assert("broadcast row jacobian rows fail",
              cmp_int_array(bcast->jacobian->p, expected_p, 4));
    mu_assert("broadcast row jacobian cols fail",
              cmp_int_array(bcast->jacobian->i, expected_i, 6));

    free_expr(bcast);
    return 0;
}

const char *test_broadcast_col_jacobian(void)
{
    /* Test jacobian of broadcast column: (3, 1) -> (3, 2)
     * Input variable: 3 elements (d1=3, d2=1)
     * Output: 6 elements (d1=3, d2=2) stored columnwise
     *
     * For a column broadcast, the column is repeated n times.
     * Child jacobian (3x1 sparsity):
     *   var[0] -> child[0]
     *   var[1] -> child[1]
     *   var[2] -> child[2]
     *
     * Broadcast column (3,1) -> (3,2) repeats the column 2 times:
     * Output in columnwise order:
     *   col 0: out[0]=child[0], out[1]=child[1], out[2]=child[2]
     *   col 1: out[3]=child[0], out[4]=child[1], out[5]=child[2]
     *
     * Broadcast jacobian (6x1 sparsity):
     *   var[0] -> out[0], out[3]
     *   var[1] -> out[1], out[4]
     *   var[2] -> out[2], out[5]
     */
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *bcast = new_broadcast(var, 3, 2);
    bcast->forward(bcast, u);
    bcast->jacobian_init(bcast);
    bcast->eval_jacobian(bcast);

    /* Each variable affects 2 elements (n times) */
    double expected_x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 1, 2, 0, 1, 2};

    mu_assert("broadcast col jacobian vals fail",
              cmp_double_array(bcast->jacobian->x, expected_x, 6));
    mu_assert("broadcast col jacobian rows fail",
              cmp_int_array(bcast->jacobian->p, expected_p, 7));
    mu_assert("broadcast col jacobian cols fail",
              cmp_int_array(bcast->jacobian->i, expected_i, 6));

    free_expr(bcast);
    return 0;
}

const char *test_broadcast_scalar_to_matrix_jacobian(void)
{
    /* Test jacobian of broadcast scalar: (1, 1) -> (2, 3)
     * Input variable: 1 element (scalar)
     * Output: 6 elements (2x3 matrix) stored columnwise
     *
     * Scalar broadcast replicates the single input value to all m*n outputs.
     * Child jacobian (6x1 sparsity): var[0] -> child[0]
     *
     * Broadcast scalar (1,1) -> (2,3) replicates in both dimensions:
     * Output in columnwise order:
     *   col 0: out[0]=var[0], out[1]=var[0]
     *   col 1: out[2]=var[0], out[3]=var[0]
     *   col 2: out[4]=var[0], out[5]=var[0]
     *
     * Broadcast jacobian (6x1 sparsity):
     *   var[0] -> out[0], out[1], out[2], out[3], out[4], out[5]
     */
    double u[1] = {5.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *bcast = new_broadcast(var, 2, 3);
    bcast->forward(bcast, u);
    bcast->jacobian_init(bcast);
    bcast->eval_jacobian(bcast);

    /* All 6 elements depend on the single input variable */
    double expected_x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 0, 0, 0, 0, 0};

    mu_assert("broadcast scalar jacobian vals fail",
              cmp_double_array(bcast->jacobian->x, expected_x, 6));
    mu_assert("broadcast scalar jacobian rows fail",
              cmp_int_array(bcast->jacobian->p, expected_p, 7));
    mu_assert("broadcast scalar jacobian cols fail",
              cmp_int_array(bcast->jacobian->i, expected_i, 6));

    free_expr(bcast);
    return 0;
}

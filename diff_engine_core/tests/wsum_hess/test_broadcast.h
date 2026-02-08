#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_broadcast_row()
{
    /* Test: wsum_hess of broadcast_row(log(x)) where x is (1, 3)
     * x = [1.0, 2.0, 4.0] (row vector)
     * broadcast to (2, 3):
     * [[1.0, 2.0, 4.0],
     *  [1.0, 2.0, 4.0]]
     *
     * log(x) is affine in the Hessian sense, log(broadcast(x)) has Hessian
     * from log applied element-wise.
     *
     * The weights are for the output (2x3 matrix), stored columnwise.
     * Weights w = [w00, w10, w01, w11, w02, w12] where w_ij is for row i, col j
     * The hessian should sum weights from replicated rows.
     */
    double x[3] = {1.0, 2.0, 4.0};
    expr *x_node = new_variable(1, 3, 0, 3);
    expr *log_node = new_log(x_node);
    expr *bcast = new_broadcast(log_node, 2, 3);

    bcast->forward(bcast, x);
    bcast->jacobian_init(bcast);
    bcast->wsum_hess_init(bcast);

    /* Weights for the 2x3 output (columnwise):
     * w = [1.0, 0.5,  2.0, 1.0,  0.25, 0.125]
     *      col0      col1         col2
     */
    double w[6] = {1.0, 0.5, 2.0, 1.0, 0.25, 0.125};
    bcast->eval_wsum_hess(bcast, w);

    /* For broadcast_row, weights are summed across the m replicas:
     * Accumulated weights for log(x):
     * w_acc[0] = w[0] + w[1] = 1.0 + 0.5 = 1.5
     * w_acc[1] = w[2] + w[3] = 2.0 + 1.0 = 3.0
     * w_acc[2] = w[4] + w[5] = 0.25 + 0.125 = 0.375
     *
     * Hessian of log(x) is diagonal with -w_acc[i] / x[i]^2:
     * H[0,0] = -1.5 / 1.0^2 = -1.5
     * H[1,1] = -3.0 / 2.0^2 = -0.75
     * H[2,2] = -0.375 / 4.0^2 = -0.0234375
     */
    double expected_x[3] = {-1.5, -0.75, -0.0234375};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("broadcast row wsum_hess: x values fail",
              cmp_double_array(bcast->wsum_hess->x, expected_x, 3));
    mu_assert("broadcast row wsum_hess: row pointers fail",
              cmp_int_array(bcast->wsum_hess->p, expected_p, 4));
    mu_assert("broadcast row wsum_hess: column indices fail",
              cmp_int_array(bcast->wsum_hess->i, expected_i, 3));

    free_expr(bcast);
    return 0;
}

const char *test_wsum_hess_broadcast_col()
{
    /* Test: wsum_hess of broadcast_col(log(x)) where x is (3, 1)
     * x = [1.0, 2.0, 4.0]^T (column vector)
     * broadcast to (3, 2):
     * [[1.0, 1.0],
     *  [2.0, 2.0],
     *  [4.0, 4.0]]
     *
     * The weights are for the output (3x2 matrix), stored columnwise.
     * Weights w = [w00, w10, w20, w01, w11, w21] where w_ij is for row i, col j
     * The hessian should sum weights from replicated columns.
     */
    double x[3] = {1.0, 2.0, 4.0};
    expr *x_node = new_variable(3, 1, 0, 3);
    expr *log_node = new_log(x_node);
    expr *bcast = new_broadcast(log_node, 3, 2);

    bcast->forward(bcast, x);
    bcast->jacobian_init(bcast);
    bcast->wsum_hess_init(bcast);

    /* Weights for the 3x2 output (columnwise):
     * w = [1.0, 0.5, 0.25,  2.0, 1.0, 0.5]
     *      col0            col1
     */
    double w[6] = {1.0, 0.5, 0.25, 2.0, 1.0, 0.5};
    bcast->eval_wsum_hess(bcast, w);

    /* For broadcast_col, weights are summed across the n replicas:
     * Accumulated weights for log(x):
     * w_acc[0] = w[0] + w[3] = 1.0 + 2.0 = 3.0
     * w_acc[1] = w[1] + w[4] = 0.5 + 1.0 = 1.5
     * w_acc[2] = w[2] + w[5] = 0.25 + 0.5 = 0.75
     *
     * Hessian of log(x) is diagonal with -w_acc[i] / x[i]^2:
     * H[0,0] = -3.0 / 1.0^2 = -3.0
     * H[1,1] = -1.5 / 2.0^2 = -0.375
     * H[2,2] = -0.75 / 4.0^2 = -0.046875
     */
    double expected_x[3] = {-3.0, -0.375, -0.046875};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("broadcast col wsum_hess: x values fail",
              cmp_double_array(bcast->wsum_hess->x, expected_x, 3));
    mu_assert("broadcast col wsum_hess: row pointers fail",
              cmp_int_array(bcast->wsum_hess->p, expected_p, 4));
    mu_assert("broadcast col wsum_hess: column indices fail",
              cmp_int_array(bcast->wsum_hess->i, expected_i, 3));

    free_expr(bcast);
    return 0;
}

const char *test_wsum_hess_broadcast_scalar_to_matrix()
{
    /* Test: wsum_hess of broadcast_scalar(log(x)) where x is scalar (1, 1)
     * x = 5.0
     * broadcast to (2, 3):
     * [[5.0, 5.0, 5.0],
     *  [5.0, 5.0, 5.0]]
     *
     * The weights are for the output (2x3 matrix), stored columnwise.
     * Weights w has 6 elements corresponding to all positions.
     * The hessian should sum all weights into the scalar weight.
     */
    double x[1] = {5.0};
    expr *x_node = new_variable(1, 1, 0, 1);
    expr *log_node = new_log(x_node);
    expr *bcast = new_broadcast(log_node, 2, 3);

    bcast->forward(bcast, x);
    bcast->jacobian_init(bcast);
    bcast->wsum_hess_init(bcast);

    /* Weights for the 2x3 output (columnwise):
     * w = [1.0, 0.5, 2.0, 1.0, 0.25, 0.125]
     */
    double w[6] = {1.0, 0.5, 2.0, 1.0, 0.25, 0.125};
    bcast->eval_wsum_hess(bcast, w);

    /* For broadcast_scalar, all weights are summed:
     * w_acc[0] = sum(w) = 1.0 + 0.5 + 2.0 + 1.0 + 0.25 + 0.125 = 4.875
     *
     * Hessian of log(scalar) is -w_acc[0] / x[0]^2:
     * H[0,0] = -4.875 / 5.0^2 = -4.875 / 25.0 = -0.195
     */
    double expected_x[1] = {-0.195};
    int expected_p[2] = {0, 1};
    int expected_i[1] = {0};

    mu_assert("broadcast scalar wsum_hess: x values fail",
              cmp_double_array(bcast->wsum_hess->x, expected_x, 1));
    mu_assert("broadcast scalar wsum_hess: row pointers fail",
              cmp_int_array(bcast->wsum_hess->p, expected_p, 2));
    mu_assert("broadcast scalar wsum_hess: column indices fail",
              cmp_int_array(bcast->wsum_hess->i, expected_i, 1));

    free_expr(bcast);
    return 0;
}

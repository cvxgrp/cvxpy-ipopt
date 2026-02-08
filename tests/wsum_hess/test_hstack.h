#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_hstack()
{
    /* Test: hstack([log(x), log(z), exp(x), sin(y)])
     * Variables: x at idx 0, z at idx 3, y at idx 6
     * x = [1, 2, 3], z = [4, 5, 6], y = [7, 8, 9]
     * Total 9 variables
     * hStacked vectorized output is 12x1: [log(x),
     *                                      log(z),
     *                                      exp(x),
     *                                      sin(y)]
     * w = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
     */

    double u_vals[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double w[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    expr *x = new_variable(3, 1, 0, 9);
    expr *z = new_variable(3, 1, 3, 9);
    expr *y = new_variable(3, 1, 6, 9);

    expr *log_x = new_log(x);
    expr *log_z = new_log(z);
    expr *exp_x = new_exp(x);
    expr *sin_y = new_sin(y);

    expr *args[4] = {log_x, log_z, exp_x, sin_y};
    expr *hstack_node = new_hstack(args, 4, 9);

    hstack_node->forward(hstack_node, u_vals);
    hstack_node->jacobian_init(hstack_node);
    hstack_node->wsum_hess_init(hstack_node);
    hstack_node->eval_wsum_hess(hstack_node, w);

    /* Expected Hessian:
     * log(x): d²/dx² = -1/x²
     *   w[0] * (-1/1²) = 1 * (-1) = -1 at (0,0)
     *   w[1] * (-1/2²) = 2 * (-0.25) = -0.5 at (1,1)
     *   w[2] * (-1/3²) = 3 * (-1/9) = -1/3 at (2,2)
     *
     * log(z): d²/dz² = -1/z²
     *   w[3] * (-1/4²) = 4 * (-1/16) = -0.25 at (3,3)
     *   w[4] * (-1/5²) = 5 * (-1/25) = -0.2 at (4,4)
     *   w[5] * (-1/6²) = 6 * (-1/36) = -1/6 at (5,5)
     *
     * exp(x): d²/dx² = exp(x)
     *   w[6] * exp(1) = 7 * e at (0,0)
     *   w[7] * exp(2) = 8 * e² at (1,1)
     *   w[8] * exp(3) = 9 * e³ at (2,2)
     *
     * sin(y): d²/dy² = -sin(y)
     *   w[9] * (-sin(7)) at (6,6)
     *   w[10] * (-sin(8)) at (7,7)
     *   w[11] * (-sin(9)) at (8,8)
     *
     * Accumulated:
     *   (0,0): -1 + 7*e
     *   (1,1): -0.5 + 8*e²
     *   (2,2): -1/3 + 9*e³
     *   (3,3): -0.25
     *   (4,4): -0.2
     *   (5,5): -1/6
     *   (6,6): -10*sin(7)
     *   (7,7): -11*sin(8)
     *   (8,8): -12*sin(9)
     */

    double e = exp(1.0);
    double e2 = exp(2.0);
    double e3 = exp(3.0);

    double expected_x[9] = {-1.0 + 7.0 * e,
                            -0.5 + 8.0 * e2,
                            -1.0 / 3.0 + 9.0 * e3,
                            -0.25,
                            -0.2,
                            -1.0 / 6.0,
                            -10.0 * sin(7.0),
                            -11.0 * sin(8.0),
                            -12.0 * sin(9.0)};

    int expected_p[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected_i[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    mu_assert("vals incorrect",
              cmp_double_array(hstack_node->wsum_hess->x, expected_x, 9));
    mu_assert("rows incorrect",
              cmp_int_array(hstack_node->wsum_hess->p, expected_p, 10));
    mu_assert("cols incorrect",
              cmp_int_array(hstack_node->wsum_hess->i, expected_i, 9));

    free_expr(hstack_node);
    return 0;
}

const char *test_wsum_hess_hstack_matrix()
{
    /* Test: hstack([log(x), log(z), exp(x), sin(y)]) with matrix variables
     * Variables: x at idx 0, z at idx 6, y at idx 12
     * Each is 3x2, so 6 elements per variable
     * x = [1 4]  z = [7  10]  y = [13 16]
     *     [2 5]      [8  11]      [14 17]
     *     [3 6]      [9  12]      [15 18]
     * Vectorized column-wise: x = [1,2,3,4,5,6], z = [7,8,9,10,11,12], y =
     * [13,14,15,16,17,18] Total 18 variables Stacked output is 24x1: [log(x),
     * log(z), exp(x), sin(y)] each 6x1 w = [1, 2, 3, ..., 24]
     */

    double u_vals[18] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                         10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
    double w[24];
    for (int i = 0; i < 24; i++)
    {
        w[i] = i + 1.0;
    }

    expr *x = new_variable(3, 2, 0, 18);
    expr *z = new_variable(3, 2, 6, 18);
    expr *y = new_variable(3, 2, 12, 18);

    expr *log_x = new_log(x);
    expr *log_z = new_log(z);
    expr *exp_x = new_exp(x);
    expr *sin_y = new_sin(y);

    expr *args[4] = {log_x, log_z, exp_x, sin_y};
    expr *hstack_node = new_hstack(args, 4, 18);

    hstack_node->forward(hstack_node, u_vals);
    hstack_node->wsum_hess_init(hstack_node);
    hstack_node->eval_wsum_hess(hstack_node, w);

    /* Expected Hessian (diagonal):
     * log(x): w[0:5] * (-1/x[0:5]²) at indices 0-5
     * log(z): w[6:11] * (-1/z[0:5]²) at indices 6-11
     * exp(x): w[12:17] * exp(x[0:5]) at indices 0-5 (accumulates with log(x))
     * sin(y): w[18:23] * (-sin(y[0:5])) at indices 12-17
     *
     * For x indices (0-5):
     *   i=0: -1/1² + 13*e¹ = -1 + 13*e
     *   i=1: -2/2² + 14*e² = -0.5 + 14*e²
     *   i=2: -3/3² + 15*e³ = -1/3 + 15*e³
     *   i=3: -4/4² + 16*e⁴ = -0.25 + 16*e⁴
     *   i=4: -5/5² + 17*e⁵ = -0.2 + 17*e⁵
     *   i=5: -6/6² + 18*e⁶ = -1/6 + 18*e⁶
     *
     * For z indices (6-11):
     *   i=0: -7/7² = -1/7
     *   i=1: -8/8² = -1/8
     *   i=2: -9/9² = -1/9
     *   i=3: -10/10² = -0.1
     *   i=4: -11/11² = -11/121
     *   i=5: -12/12² = -1/12
     *
     * For y indices (12-17):
     *   i=0: -19*sin(13)
     *   i=1: -20*sin(14)
     *   i=2: -21*sin(15)
     *   i=3: -22*sin(16)
     *   i=4: -23*sin(17)
     *   i=5: -24*sin(18)
     */

    double expected_x[18];
    // x indices (0-5) - accumulation of log and exp
    expected_x[0] = -1.0 + 13.0 * exp(1.0);
    expected_x[1] = -0.5 + 14.0 * exp(2.0);
    expected_x[2] = -1.0 / 3.0 + 15.0 * exp(3.0);
    expected_x[3] = -0.25 + 16.0 * exp(4.0);
    expected_x[4] = -0.2 + 17.0 * exp(5.0);
    expected_x[5] = -1.0 / 6.0 + 18.0 * exp(6.0);

    // z indices (6-11) - only log
    expected_x[6] = -1.0 / 7.0;
    expected_x[7] = -1.0 / 8.0;
    expected_x[8] = -1.0 / 9.0;
    expected_x[9] = -0.1;
    expected_x[10] = -11.0 / 121.0;
    expected_x[11] = -1.0 / 12.0;

    // y indices (12-17) - only sin
    expected_x[12] = -19.0 * sin(13.0);
    expected_x[13] = -20.0 * sin(14.0);
    expected_x[14] = -21.0 * sin(15.0);
    expected_x[15] = -22.0 * sin(16.0);
    expected_x[16] = -23.0 * sin(17.0);
    expected_x[17] = -24.0 * sin(18.0);

    int expected_p[19] = {0,  1,  2,  3,  4,  5,  6,  7,  8, 9,
                          10, 11, 12, 13, 14, 15, 16, 17, 18};
    int expected_i[18] = {0, 1,  2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16, 17};

    mu_assert("vals incorrect",
              cmp_double_array(hstack_node->wsum_hess->x, expected_x, 18));
    mu_assert("rows incorrect",
              cmp_int_array(hstack_node->wsum_hess->p, expected_p, 19));
    mu_assert("cols incorrect",
              cmp_int_array(hstack_node->wsum_hess->i, expected_i, 18));

    free_expr(hstack_node);
    return 0;
}

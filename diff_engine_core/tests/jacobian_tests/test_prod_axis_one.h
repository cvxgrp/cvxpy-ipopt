#include <stdio.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_jacobian_prod_axis_one()
{
    /* x is 3x3 variable, global index 1, total 10 vars
     * x = [1, 2, 3, 4, 5, 6, 7, 8, 9] (column-major order)
     *     [1, 4, 7]
     *     [2, 5, 8]
     *     [3, 6, 9]
     *
     * f = prod_axis_one(x) = [28, 80, 162] (row vector)
     * Row 0: 1*4*7 = 28
     * Row 1: 2*5*8 = 80
     * Row 2: 3*6*9 = 162
     *
     * Jacobian is 3x10 (3 outputs, 10 total vars):
     * Row 0 (output[0] = 28):  df[0]/dx = [0, 28, 0, 0, 7, 0, 0, 4, 0, 0]
     *   df[0]/dx[0] = 28/1 = 28, df[0]/dx[3] = 28/4 = 7, df[0]/dx[6] = 28/7 = 4
     * Row 1 (output[1] = 80): df[1]/dx = [0, 0, 40, 0, 0, 16, 0, 0, 10, 0]
     *   df[1]/dx[1] = 80/2 = 40, df[1]/dx[4] = 80/5 = 16, df[1]/dx[7] = 80/8 = 10
     * Row 2 (output[2] = 162): df[2]/dx = [0, 0, 0, 54, 0, 0, 27, 0, 0, 18]
     *   df[2]/dx[2] = 162/3 = 54, df[2]/dx[5] = 162/6 = 27, df[2]/dx[8] = 162/9 = 18
     *
     * (Note: global indices are offset by var_id=1)
     */
    double u_vals[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    expr *x = new_variable(3, 3, 1, 10);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->jacobian_init(p);
    p->eval_jacobian(p);

    /* CSR format for 3x10 Jacobian with row-strided structure */
    double expected_Ax[9] = {28.0, 7.0, 4.0, 40.0, 16.0, 10.0, 54.0, 27.0, 18.0};
    int expected_Ap[4] = {0, 3, 6, 9};
    int expected_Ai[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    mu_assert("vals fail", cmp_double_array(p->jacobian->x, expected_Ax, 9));
    mu_assert("rows fail", cmp_int_array(p->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(p->jacobian->i, expected_Ai, 9));

    free_expr(p);
    return 0;
}

const char *test_jacobian_prod_axis_one_one_zero()
{
    /* x is 3x3 variable, global index 1, total 10 vars
     * x = [1, 2, 3, 4, 0, 6, 7, 8, 9] (column-major order)
     *     [1, 4, 7]
     *     [2, 0, 8]
     *     [3, 6, 9]
     *
     * f = prod_axis_one(x) = [28, 0, 162] (row vector)
     * Row 0: 1*4*7 = 28
     * Row 1: 2*0*8 = 0 (one zero at column 1)
     * Row 2: 3*6*9 = 162
     *
     * Jacobian is 3x10:
     * Row 0 (output[0] = 28, no zeros): df[0]/dx = [0, 28, 0, 0, 7, 0, 0, 4, 0, 0]
     *   df[0]/dx[0] = 28/1 = 28, df[0]/dx[3] = 28/4 = 7, df[0]/dx[6] = 28/7 = 4
     * Row 1 (output[1] = 0, one zero): df[1]/dx = [0, 0, 0, 0, 0, 16, 0, 0, 0, 0]
     *   Only derivative w.r.t. zero element: df[1]/dx[4] = prod_nonzero = 2*8 = 16
     * Row 2 (output[2] = 162, no zeros): df[2]/dx = [0, 0, 0, 54, 0, 0, 27, 0, 0,
     * 18] df[2]/dx[2] = 162/3 = 54, df[2]/dx[5] = 162/6 = 27, df[2]/dx[8] = 162/9 =
     * 18
     */
    double u_vals[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0};
    expr *x = new_variable(3, 3, 1, 10);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->jacobian_init(p);
    p->eval_jacobian(p);

    /* CSR format for 3x10 Jacobian with row-strided structure */
    double expected_Ax[9] = {28.0, 7.0, 4.0, 0.0, 16.0, 0.0, 54.0, 27.0, 18.0};
    int expected_Ap[4] = {0, 3, 6, 9};
    int expected_Ai[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    mu_assert("vals fail", cmp_double_array(p->jacobian->x, expected_Ax, 9));
    mu_assert("rows fail", cmp_int_array(p->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(p->jacobian->i, expected_Ai, 9));

    free_expr(p);
    return 0;
}

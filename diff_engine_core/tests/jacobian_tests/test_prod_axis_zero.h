#include <stdio.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_jacobian_prod_axis_zero()
{
    /* x is 2x3 variable, global index 1, total 8 vars
     * x = [1, 2, 3, 4, 5, 6] (column-major order)
     *     [1, 3, 5]
     *     [2, 4, 6]
     *
     * f = prod_axis_zero(x) = [2, 12, 30]
     *
     * Jacobian is 3x8:
     * Row 0 (output[0] = 2):  df[0]/dx = [2, 1, 0, 0, 0, 0, 0, 0]
     *   df[0]/dx[0] = 2/1 = 2, df[0]/dx[1] = 2/2 = 1
     * Row 1 (output[1] = 12): df[1]/dx = [0, 0, 4, 3, 0, 0, 0, 0]
     *   df[1]/dx[2] = 12/3 = 4, df[1]/dx[3] = 12/4 = 3
     * Row 2 (output[2] = 30): df[2]/dx = [0, 0, 0, 0, 6, 5, 0, 0]
     *   df[2]/dx[4] = 30/5 = 6, df[2]/dx[5] = 30/6 = 5
     */
    double u_vals[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    expr *x = new_variable(2, 3, 1, 8);
    expr *p = new_prod_axis_zero(x);

    p->forward(p, u_vals);
    p->jacobian_init(p);
    p->eval_jacobian(p);

    /* CSR format for 3x8 Jacobian with block diagonal structure */
    double expected_Ax[6] = {2.0, 1.0, 4.0, 3.0, 6.0, 5.0};
    int expected_Ap[4] = {0, 2, 4, 6};
    int expected_Ai[6] = {1, 2, 3, 4, 5, 6};

    mu_assert("vals fail", cmp_double_array(p->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(p->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(p->jacobian->i, expected_Ai, 6));

    free_expr(p);
    return 0;
}

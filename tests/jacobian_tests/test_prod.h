#include <stdio.h>

#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_jacobian_prod_no_zero()
{
    /* x is 4x1 variable, global index 2, total 8 vars
     * x = [1, 2, 3, 4]
     * f = prod(x) = 24
     * df/dx = [24, 12, 8, 6]
     */
    double u_vals[8] = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0};
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->jacobian_init(p);
    p->eval_jacobian(p);

    double expected_Ax[4] = {24.0, 12.0, 8.0, 6.0};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->jacobian->x, expected_Ax, 4));
    mu_assert("rows fail", cmp_int_array(p->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(p->jacobian->i, expected_Ai, 4));

    free_expr(p);
    return 0;
}

const char *test_jacobian_prod_one_zero()
{
    /* x = [1, 0, 3, 4], zero at index 1
     * df/dx = [0, prod_nonzero, 0, 0] = [0, 12, 0, 0]
     */
    double u_vals[8] = {0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0};
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->jacobian_init(p);
    p->eval_jacobian(p);

    double expected_Ax[4] = {0.0, 12.0, 0.0, 0.0};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->jacobian->x, expected_Ax, 4));
    mu_assert("rows fail", cmp_int_array(p->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(p->jacobian->i, expected_Ai, 4));

    free_expr(p);
    return 0;
}

const char *test_jacobian_prod_two_zeros()
{
    /* x = [1, 0, 0, 4], two zeros -> Jacobian all zeros */
    double u_vals[8] = {0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0};
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->jacobian_init(p);
    p->eval_jacobian(p);

    double expected_Ax[4] = {0.0, 0.0, 0.0, 0.0};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->jacobian->x, expected_Ax, 4));
    mu_assert("rows fail", cmp_int_array(p->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(p->jacobian->i, expected_Ai, 4));

    free_expr(p);
    return 0;
}

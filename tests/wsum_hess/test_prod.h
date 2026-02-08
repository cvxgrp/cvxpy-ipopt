#include <stdio.h>
#include <string.h>

#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

/* Common setup: x is 4x1 variable, global index 2, total 8 vars */

const char *test_wsum_hess_prod_no_zero()
{
    /* x = [1, 2, 3, 4], f = 24 */
    double u_vals[8] = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0};
    double w = 1.0;
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, &w);

    /* Row-major over dense 4x4 block */
    double expected_x[16] = {0.0, 12.0, 8.0, 6.0, 12.0, 0.0, 4.0, 3.0,
                             8.0, 4.0,  0.0, 2.0, 6.0,  3.0, 2.0, 0.0};

    int expected_p[9] = {0, 0, 0, 4, 8, 12, 16, 16, 16};
    int expected_i[16] = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 16));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 16));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_one_zero()
{
    /* x = [1, 0, 3, 4], zero at index 1, prod_nonzero = 12 */
    double u_vals[8] = {0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0};
    double w = 1.0;
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, &w);

    double expected_x[16];
    memset(expected_x, 0, sizeof(expected_x));
    /* Row 1 / col 1 nonzeros */
    expected_x[1 * 4 + 0] = 12.0; /* (1,0) */
    expected_x[1 * 4 + 2] = 4.0;  /* (1,2) */
    expected_x[1 * 4 + 3] = 3.0;  /* (1,3) */
    expected_x[0 * 4 + 1] = 12.0; /* symmetric */
    expected_x[2 * 4 + 1] = 4.0;
    expected_x[3 * 4 + 1] = 3.0;

    int expected_p[9] = {0, 0, 0, 4, 8, 12, 16, 16, 16};
    int expected_i[16] = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 16));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 16));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_two_zeros()
{
    /* x = [1, 0, 0, 4], zeros at 1 and 2, prod over others = 4 */
    double u_vals[8] = {0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0};
    double w = 1.0;
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, &w);

    double expected_x[16];
    memset(expected_x, 0, sizeof(expected_x));
    expected_x[1 * 4 + 2] = 4.0; /* (1,2) */
    expected_x[2 * 4 + 1] = 4.0; /* (2,1) */

    int expected_p[9] = {0, 0, 0, 4, 8, 12, 16, 16, 16};
    int expected_i[16] = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 16));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 16));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_many_zeros()
{
    /* x = [0, 0, 0, 4], three zeros => Hessian all zeros */
    double u_vals[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
    double w = 1.0;
    expr *x = new_variable(4, 1, 2, 8);
    expr *p = new_prod(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, &w);

    double expected_x[16];
    memset(expected_x, 0, sizeof(expected_x));

    int expected_p[9] = {0, 0, 0, 4, 8, 12, 16, 16, 16};
    int expected_i[16] = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 16));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 16));

    free_expr(p);
    return 0;
}

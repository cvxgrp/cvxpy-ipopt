#ifndef TEST_WSUM_HESS_TRANSPOSE_H
#define TEST_WSUM_HESS_TRANSPOSE_H

#include "affine.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_transpose()
{

    expr *X = new_variable(2, 2, 0, 8);
    expr *Y = new_variable(2, 2, 4, 8);

    expr *XY = new_matmul(X, Y);
    expr *XYT = new_transpose(XY);

    double u[8] = {1, 3, 2, 4, 5, 7, 6, 8};
    XYT->forward(XYT, u);
    XYT->wsum_hess_init(XYT);
    double w[4] = {1, 2, 3, 4};
    XYT->eval_wsum_hess(XYT, w);

    double expected_x[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 1, 3, 2, 4, 2, 4};
    int expected_p[9] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    int expected_i[16] = {4, 6, 4, 6, 5, 7, 5, 7, 0, 1, 2, 3, 0, 1, 2, 3};

    mu_assert("hess values fail",
              cmp_double_array(XYT->wsum_hess->x, expected_x, 8));
    mu_assert("jacobian row ptr fail",
              cmp_int_array(XYT->wsum_hess->p, expected_p, 5));
    mu_assert("jacobian col idx fail",
              cmp_int_array(XYT->wsum_hess->i, expected_i, 8));
    free_expr(XYT);

    return 0;
}

#endif // TEST_WSUM_HESS_TRANSPOSE_H

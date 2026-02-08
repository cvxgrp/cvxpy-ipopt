#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_matmul()
{
    /* Test: Z = X @ Y where X is 2x3, Y is 3x4
     * var = (X, Y) where X starts at index 0 (size 6), Y starts at index 6 (size 12)
     * Total n_vars = 18
     * Z is 2x4 (size 8)
     */

    int m = 2, k = 3, n = 4;
    int x_size = m * k;           // 6
    int y_size = k * n;           // 12
    int n_vars = x_size + y_size; // 18

    /* X values (2x3) in column-major order */
    double x_vals[6] = {1.0, 2.0,  /* col 0 */
                        3.0, 4.0,  /* col 1 */
                        5.0, 6.0}; /* col 2 */

    /* Y values (3x4) in column-major order */
    double y_vals[12] = {7.0,  8.0,  9.0,   /* col 0 */
                         10.0, 11.0, 12.0,  /* col 1 */
                         13.0, 14.0, 15.0,  /* col 2 */
                         16.0, 17.0, 18.0}; /* col 3 */

    /* Combined parameter vector */
    double u_vals[18];
    for (int i = 0; i < 6; i++) u_vals[i] = x_vals[i];
    for (int i = 0; i < 12; i++) u_vals[6 + i] = y_vals[i];

    /* Weights for the 8 outputs */
    double w[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    /* Create variables */
    expr *X = new_variable(m, k, 0, n_vars);
    expr *Y = new_variable(k, n, x_size, n_vars);
    expr *Z = new_matmul(X, Y);

    /* Forward pass and Hessian initialization */
    Z->forward(Z, u_vals);
    Z->wsum_hess_init(Z);
    Z->eval_wsum_hess(Z, w);

    /* Verify Hessian dimensions and sparsity */
    mu_assert("Hessian should be 18x18", Z->wsum_hess->m == n_vars);
    mu_assert("Hessian should be 18x18", Z->wsum_hess->n == n_vars);
    mu_assert("Hessian should have 48 nonzeros", Z->wsum_hess->nnz == 48);

    int expected_p[19] = {0,  4,  8,  12, 16, 20, 24, 26, 28, 30,
                          32, 34, 36, 38, 40, 42, 44, 46, 48};

    mu_assert("Row pointers incorrect",
              cmp_int_array(Z->wsum_hess->p, expected_p, 19));

    int expected_i[48] = {6, 9,  12, 15, /* row 0 */
                          6, 9,  12, 15, /* row 1 */
                          7, 10, 13, 16, /* row 2 */
                          7, 10, 13, 16, /* row 3 */
                          8, 11, 14, 17, /* row 4 */
                          8, 11, 14, 17, /* row 5 */
                          0, 1,          /* row 6*/
                          2, 3,          /* row 7*/
                          4, 5,          /* row 8*/
                          0, 1,          /* row 9*/
                          2, 3,          /* row 10*/
                          4, 5,          /* row 11*/
                          0, 1,          /* row 12*/
                          2, 3,          /* row 13*/
                          4, 5,          /* row 14*/
                          0, 1,          /* row 15*/
                          2, 3,          /* row 16*/
                          4, 5};

    mu_assert("Column indices incorrect",
              cmp_int_array(Z->wsum_hess->i, expected_i, 48));

    double expected_x[48] = {1.0, 3.0, 5.0, 7.0, /* row 0 */
                             2.0, 4.0, 6.0, 8.0, /* row 1 */
                             1.0, 3.0, 5.0, 7.0, /* row 2 */
                             2.0, 4.0, 6.0, 8.0, /* row 3 */
                             1.0, 3.0, 5.0, 7.0, /* row 4 */
                             2.0, 4.0, 6.0, 8.0, /* row 5 */
                             1.0, 2.0,           /* row 6 */
                             1.0, 2.0,           /* row 7 */
                             1.0, 2.0,           /* row 8 */
                             3.0, 4.0,           /* row 9 */
                             3.0, 4.0,           /* row 10 */
                             3.0, 4.0,           /* row 11 */
                             5.0, 6.0,           /* row 12 */
                             5.0, 6.0,           /* row 13 */
                             5.0, 6.0,           /* row 14 */
                             7.0, 8.0,           /* row 15 */
                             7.0, 8.0,           /* row 16 */
                             7.0, 8.0};          /* row 17 */

    mu_assert("Hessian values incorrect",
              cmp_double_array(Z->wsum_hess->x, expected_x, 48));

    free_expr(Z);
    return 0;
}

const char *test_wsum_hess_matmul_yx()
{
    /* Test: Z = X @ Y where X is 2x3, Y is 3x4
     * var = (Y, X) where Y starts at index 0 (size 12), X starts at index 12 (size
     * 6) Total n_vars = 18 Z is 2x4 (size 8)
     */

    int m = 2, k = 3, n = 4;
    int y_size = k * n;           // 12
    int x_size = m * k;           // 6
    int n_vars = x_size + y_size; // 18

    /* X values (2x3) in column-major order */
    double x_vals[6] = {1.0, 2.0,  /* col 0 */
                        3.0, 4.0,  /* col 1 */
                        5.0, 6.0}; /* col 2 */

    /* Y values (3x4) in column-major order */
    double y_vals[12] = {7.0,  8.0,  9.0,   /* col 0 */
                         10.0, 11.0, 12.0,  /* col 1 */
                         13.0, 14.0, 15.0,  /* col 2 */
                         16.0, 17.0, 18.0}; /* col 3 */

    /* Combined parameter vector: Y first, then X */
    double u_vals[18];
    for (int i = 0; i < 12; i++) u_vals[i] = y_vals[i];
    for (int i = 0; i < 6; i++) u_vals[12 + i] = x_vals[i];

    /* Weights for the 8 outputs */
    double w[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    /* Create variables with Y first */
    expr *Y = new_variable(k, n, 0, n_vars);
    expr *X = new_variable(m, k, y_size, n_vars);
    expr *Z = new_matmul(X, Y);

    /* Forward pass and Hessian initialization */
    Z->forward(Z, u_vals);
    Z->wsum_hess_init(Z);
    Z->eval_wsum_hess(Z, w);

    /* Verify Hessian dimensions and sparsity */
    mu_assert("Hessian should be 18x18", Z->wsum_hess->m == n_vars);
    mu_assert("Hessian should be 18x18", Z->wsum_hess->n == n_vars);
    mu_assert("Hessian should have 48 nonzeros", Z->wsum_hess->nnz == 48);

    /* Row pointers when Y < X:
     * Rows 0-11 (Y variables): each couples with m=2 X variables
     * Rows 12-17 (X variables): each couples with n=4 Y variables
     */
    int expected_p[19] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18,
                          20, 22, 24, 28, 32, 36, 40, 44, 48};

    mu_assert("Row pointers incorrect",
              cmp_int_array(Z->wsum_hess->p, expected_p, 19));

    /* Column indices when Y < X:
     * Y[k_idx, col] couples with X[row, k_idx] for all row
     * X variable index = 12 + row + k_idx*m
     * X[row, k_idx] couples with Y[k_idx, col] for all col
     * Y variable index = 0 + k_idx + col*k
     */
    int expected_i[48] = {12, 13,         /* row 0: Y[0,0] */
                          14, 15,         /* row 1: Y[1,0] */
                          16, 17,         /* row 2: Y[2,0] */
                          12, 13,         /* row 3: Y[0,1] */
                          14, 15,         /* row 4: Y[1,1] */
                          16, 17,         /* row 5: Y[2,1] */
                          12, 13,         /* row 6: Y[0,2] */
                          14, 15,         /* row 7: Y[1,2] */
                          16, 17,         /* row 8: Y[2,2] */
                          12, 13,         /* row 9: Y[0,3] */
                          14, 15,         /* row 10: Y[1,3] */
                          16, 17,         /* row 11: Y[2,3] */
                          0,  3,  6, 9,   /* row 12: X[0,0] */
                          0,  3,  6, 9,   /* row 13: X[1,0] */
                          1,  4,  7, 10,  /* row 14: X[0,1] */
                          1,  4,  7, 10,  /* row 15: X[1,1] */
                          2,  5,  8, 11,  /* row 16: X[0,2] */
                          2,  5,  8, 11}; /* row 17: X[1,2] */

    mu_assert("Column indices incorrect",
              cmp_int_array(Z->wsum_hess->i, expected_i, 48));

    double expected_x[48] = {1.0, 2.0,            /* row 0 */
                             1.0, 2.0,            /* row 1 */
                             1.0, 2.0,            /* row 2 */
                             3.0, 4.0,            /* row 3 */
                             3.0, 4.0,            /* row 4 */
                             3.0, 4.0,            /* row 5 */
                             5.0, 6.0,            /* row 6 */
                             5.0, 6.0,            /* row 7 */
                             5.0, 6.0,            /* row 8 */
                             7.0, 8.0,            /* row 9 */
                             7.0, 8.0,            /* row 10 */
                             7.0, 8.0,            /* row 11 */
                             1.0, 3.0, 5.0, 7.0,  /* row 12 */
                             2.0, 4.0, 6.0, 8.0,  /* row 13 */
                             1.0, 3.0, 5.0, 7.0,  /* row 14 */
                             2.0, 4.0, 6.0, 8.0,  /* row 15 */
                             1.0, 3.0, 5.0, 7.0,  /* row 16 */
                             2.0, 4.0, 6.0, 8.0}; /* row 17 */

    mu_assert("Hessian values incorrect",
              cmp_double_array(Z->wsum_hess->x, expected_x, 48));

    free_expr(Z);
    return 0;
}

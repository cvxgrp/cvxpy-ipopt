#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_jacobian_matmul()
{
    /* Test: Z = X @ Y where X is 2x3, Y is 3x4
     * var = (X, Y) where X starts at index 0 (size 6), Y starts at index 6 (size 12)
     * Total n_vars = 18
     * Z is 2x4 (size 8)
     *
     * Each row of Jacobian has 6 nonzeros (3 from X, 3 from Y)
     * Total nnz = 8 * 6 = 48
     */

    int m = 2, k = 3, n = 4;
    int x_size = m * k;           // 6
    int y_size = k * n;           // 12
    int z_size = m * n;           // 8
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

    /* Create variables */
    expr *X = new_variable(m, k, 0, n_vars);
    expr *Y = new_variable(k, n, x_size, n_vars);
    expr *Z = new_matmul(X, Y);

    /* Forward pass and jacobian initialization */
    Z->forward(Z, u_vals);
    Z->jacobian_init(Z);
    Z->eval_jacobian(Z);

    /* Verify sparsity pattern */
    mu_assert("Jacobian should have 8 rows", Z->jacobian->m == z_size);
    mu_assert("Jacobian should have 18 columns", Z->jacobian->n == n_vars);
    mu_assert("Jacobian should have 48 nonzeros", Z->jacobian->nnz == 48);

    /* Check row pointers: each row should have 6 entries */
    int expected_p[9] = {0, 6, 12, 18, 24, 30, 36, 42, 48};
    mu_assert("Row pointers incorrect",
              cmp_int_array(Z->jacobian->p, expected_p, 9));

    int expected_i[48] = {0, 2, 4, 6,  7,  8,   /* row 0 */
                          1, 3, 5, 6,  7,  8,   /* row 1 */
                          0, 2, 4, 9,  10, 11,  /* row 2 */
                          1, 3, 5, 9,  10, 11,  /* row 3 */
                          0, 2, 4, 12, 13, 14,  /* row 4 */
                          1, 3, 5, 12, 13, 14,  /* row 5 */
                          0, 2, 4, 15, 16, 17,  /* row 6 */
                          1, 3, 5, 15, 16, 17}; /* row 7 */
    mu_assert("Column indices incorrect",
              cmp_int_array(Z->jacobian->i, expected_i, 48));

    /* Verify Jacobian values row-wise: for each row, values are
       [Y^T row for the column, X row values] since X has lower var_id */
    double expected_x[48] = {
        /* row 0 (col 0) */ 7.0,  8.0,  9.0,  1.0, 3.0, 5.0,
        /* row 1 (col 0) */ 7.0,  8.0,  9.0,  2.0, 4.0, 6.0,
        /* row 2 (col 1) */ 10.0, 11.0, 12.0, 1.0, 3.0, 5.0,
        /* row 3 (col 1) */ 10.0, 11.0, 12.0, 2.0, 4.0, 6.0,
        /* row 4 (col 2) */ 13.0, 14.0, 15.0, 1.0, 3.0, 5.0,
        /* row 5 (col 2) */ 13.0, 14.0, 15.0, 2.0, 4.0, 6.0,
        /* row 6 (col 3) */ 16.0, 17.0, 18.0, 1.0, 3.0, 5.0,
        /* row 7 (col 3) */ 16.0, 17.0, 18.0, 2.0, 4.0, 6.0};

    mu_assert("Jacobian values incorrect",
              cmp_double_array(Z->jacobian->x, expected_x, 48));

    free_expr(Z);
    return 0;
}

#include <stdio.h>
#include <string.h>

#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_wsum_hess_prod_axis_one_no_zeros()
{
    /* x is 2x3 variable, global index 1, total 8 vars
     * x = [1, 2, 3, 4, 5, 6] (column-major)
     *     [1, 3, 5]
     *     [2, 4, 6]
     * f = prod_axis_one(x) = [15, 48]
     * w = [1, 2]
     *
     * Hessian is block diagonal with two 3x3 blocks (one per row):
     * Row 0 block (scale = 1 * 15 = 15):
     *   cols (c0=1, c1=3, c2=5)
     *   off-diagonals: (0,1)=5, (0,2)=3, (1,0)=5, (1,2)=1, (2,0)=3, (2,1)=1; diag=0
     * Row 1 block (scale = 2 * 48 = 96):
     *   cols (c0=2, c1=4, c2=6)
     *   off-diagonals: (0,1)=12, (0,2)=8, (1,0)=12, (1,2)=4, (2,0)=8, (2,1)=4;
     * diag=0
     */
    double u_vals[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    double w_vals[2] = {1.0, 2.0};
    expr *x = new_variable(2, 3, 1, 8);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[12] = {/* Var 1 (row 0, col 0): [5, 3] (excludes col 0) */
                             5.0, 3.0,
                             /* Var 2 (row 1, col 0): [12, 8] (excludes col 0) */
                             12.0, 8.0,
                             /* Var 3 (row 0, col 1): [5, 1] (excludes col 1) */
                             5.0, 1.0,
                             /* Var 4 (row 1, col 1): [12, 4] (excludes col 1) */
                             12.0, 4.0,
                             /* Var 5 (row 0, col 2): [3, 1] (excludes col 2) */
                             3.0, 1.0,
                             /* Var 6 (row 1, col 2): [8, 4] (excludes col 2) */
                             8.0, 4.0};

    /* Row pointers (monotonically increasing for valid CSR format) */
    int expected_p[9] = {0, 0, 2, 4, 6, 8, 10, 12, 12};

    /* Column indices (each row of the matrix interacts with its own columns,
     * excluding diagonal) */
    int expected_i[12] = {/* Var 1 (row 0, col 0): cols 3,5 (excludes 1) */
                          3, 5,
                          /* Var 2 (row 1, col 0): cols 4,6 (excludes 2) */
                          4, 6,
                          /* Var 3 (row 0, col 1): cols 1,5 (excludes 3) */
                          1, 5,
                          /* Var 4 (row 1, col 1): cols 2,6 (excludes 4) */
                          2, 6,
                          /* Var 5 (row 0, col 2): cols 1,3 (excludes 5) */
                          1, 3,
                          /* Var 6 (row 1, col 2): cols 2,4 (excludes 6) */
                          2, 4};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 12));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 12));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_one_one_zero()
{
    /* x is 3x3 variable, global index 1, total 10 vars
     * x = [1, 2, 3, 4, 0, 6, 7, 8, 9] (column-major)
     *     [1, 4, 7]
     *     [2, 0, 8]
     *     [3, 6, 9]
     * f = prod_axis_one(x) = [28, 0, 162]
     * w = [1, 2, 3]
     *
     * Blocks are 3x3 (one per row):
     * Row 0 (no zeros, scale=1*28=28):
     *   cols (1,4,7): off-diag (0,1)=7, (0,2)=4, (1,0)=7, (1,2)=2, (2,0)=4, (2,1)=2
     * Row 1 (one zero at col 1, prod_nonzero=16, scale=2*16=32):
     *   only row/col 1 nonzero: (1,0)=32/2=16, (1,2)=32/8=4 (symmetric)
     * Row 2 (no zeros, scale=3*162=486):
     *   cols (3,6,9): off-diag (0,1)=27, (0,2)=18, (1,0)=27, (1,2)=9, (2,0)=18,
     * (2,1)=9
     */
    double u_vals[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0};
    double w_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 3, 1, 10);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[18];
    memset(expected_x, 0, sizeof(expected_x));

    /* Var 1 (row 0, col 0): [7, 4] (excludes col 0) */
    expected_x[0] = 7.0;
    expected_x[1] = 4.0;

    /* Var 2 (row 1, col 0): [16, 0] (excludes col 0, one zero at col 1) */
    expected_x[2] = 16.0;
    expected_x[3] = 0.0;

    /* Var 3 (row 2, col 0): [27, 18] (excludes col 0) */
    expected_x[4] = 27.0;
    expected_x[5] = 18.0;

    /* Var 4 (row 0, col 1): [7, 1] (excludes col 1) */
    expected_x[6] = 7.0;
    expected_x[7] = 1.0;

    /* Var 5 (row 1, col 1): [16, 4] (excludes col 1, one zero at col 1) */
    expected_x[8] = 16.0;
    expected_x[9] = 4.0;

    /* Var 6 (row 2, col 1): [27, 9] (excludes col 1) */
    expected_x[10] = 27.0;
    expected_x[11] = 9.0;

    /* Var 7 (row 0, col 2): [4, 1] (excludes col 2) */
    expected_x[12] = 4.0;
    expected_x[13] = 1.0;

    /* Var 8 (row 1, col 2): [0, 4] (excludes col 2, one zero at col 1) */
    expected_x[14] = 0.0;
    expected_x[15] = 4.0;

    /* Var 9 (row 2, col 2): [18, 9] (excludes col 2) */
    expected_x[16] = 18.0;
    expected_x[17] = 9.0;

    /* Row pointers (monotonically increasing for valid CSR format) */
    int expected_p[11] = {0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18};

    /* Column indices (each row of the matrix interacts with its own columns,
     * excluding diagonal) */
    int expected_i[18] = {/* Var 1 (row 0, col 0): cols 4,7 (excludes 1) */
                          4, 7,
                          /* Var 2 (row 1, col 0): cols 5,8 (excludes 2) */
                          5, 8,
                          /* Var 3 (row 2, col 0): cols 6,9 (excludes 3) */
                          6, 9,
                          /* Var 4 (row 0, col 1): cols 1,7 (excludes 4) */
                          1, 7,
                          /* Var 5 (row 1, col 1): cols 2,8 (excludes 5) */
                          2, 8,
                          /* Var 6 (row 2, col 1): cols 3,9 (excludes 6) */
                          3, 9,
                          /* Var 7 (row 0, col 2): cols 1,4 (excludes 7) */
                          1, 4,
                          /* Var 8 (row 1, col 2): cols 2,5 (excludes 8) */
                          2, 5,
                          /* Var 9 (row 2, col 2): cols 3,6 (excludes 9) */
                          3, 6};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 18));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 11));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 18));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_one_mixed_zeros()
{
    /* x is 5x3 variable, global index 1, total 16 vars
     * Rows (axis=1 products):
     *   r0: [1, 2, 1] -> no zeros, prod=2
     *   r1: [1, 0, 0] -> two zeros (cols 1,2), prod_nonzero=1
     *   r2: [1, 3, 0] -> one zero (col 2), prod_nonzero=3
     *   r3: [1, 4, 2] -> no zeros, prod=8
     *   r4: [1, 5, 3] -> no zeros, prod=15
     * w = [1, 2, 3, 4, 5]
     * Blocks are 3x3 (one per row, block diagonal):
     */
    double u_vals[16] = {0.0,
                         /* col 0 (rows 0-4): 1,1,1,1,1 */
                         1.0, 1.0, 1.0, 1.0, 1.0,
                         /* col 1: 2,0,3,4,5 */
                         2.0, 0.0, 3.0, 4.0, 5.0,
                         /* col 2: 1,0,0,2,3 */
                         1.0, 0.0, 0.0, 2.0, 3.0};
    /* Actually store column-major:
     * col0: [1,1,1,1,1]
     * col1: [2,0,3,4,5]
     * col2: [1,0,0,2,3]
     */
    double w_vals[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    expr *x = new_variable(5, 3, 1, 16);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[30];
    memset(expected_x, 0, sizeof(expected_x));

    /* For a 5x3 matrix with var_id=1, each row has 2 nnz (d2-1):
     * CSR row pointers: p[i] = (i-1)*2 for i in [1,15]
     *   Var 1 (matrix [0,0]): p[1]=0
     *   Var 2 (matrix [1,0]): p[2]=2
     *   Var 3 (matrix [2,0]): p[3]=4
     *   Var 4 (matrix [3,0]): p[4]=6
     *   Var 5 (matrix [4,0]): p[5]=8
     *   Var 6 (matrix [0,1]): p[6]=10
     *   Var 7 (matrix [1,1]): p[7]=12
     *   Var 8 (matrix [2,1]): p[8]=14
     *   Var 9 (matrix [3,1]): p[9]=16
     *   Var 10 (matrix [4,1]): p[10]=18
     *   Var 11 (matrix [0,2]): p[11]=20
     *   Var 12 (matrix [1,2]): p[12]=22
     *   Var 13 (matrix [2,2]): p[13]=24
     *   Var 14 (matrix [3,2]): p[14]=26
     *   Var 15 (matrix [4,2]): p[15]=28
     */

    /* Row 0 block (no zeros, scale = 1 * 2 = 2), x = [1,2,1] */
    /* Var 1 (matrix [0,0], col 0): [1, 2] (excludes col 0) */
    expected_x[0] = 1.0; /* 2/(1*2) */
    expected_x[1] = 2.0; /* 2/(1*1) */

    /* Var 6 (matrix [0,1], col 1): [1, 1] (excludes col 1) */
    expected_x[10] = 1.0; /* 2/(2*1) */
    expected_x[11] = 1.0; /* 2/(2*1) */

    /* Var 11 (matrix [0,2], col 2): [2, 1] (excludes col 2) */
    expected_x[20] = 2.0; /* 2/(1*1) */
    expected_x[21] = 1.0; /* 2/(1*2) */

    /* Row 1 block (two zeros at cols 1,2), hess = w*prod_nonzero = 2*1 = 2 */
    /* Var 2 (matrix [1,0], col 0): [0, 0] (excludes col 0) */
    expected_x[2] = 0.0;
    expected_x[3] = 0.0;

    /* Var 7 (matrix [1,1], col 1): [0, 2] (excludes col 1) */
    expected_x[12] = 0.0;
    expected_x[13] = 2.0; /* (1,2) */

    /* Var 12 (matrix [1,2], col 2): [0, 2] (excludes col 2) */
    expected_x[22] = 0.0;
    expected_x[23] = 2.0; /* (2,1) */

    /* Row 2 block (one zero at col 2), w_prod = 3 * 3 = 9, x = [1,3,0] */
    /* Var 3 (matrix [2,0], col 0): [0, 9] (excludes col 0) */
    expected_x[4] = 0.0;
    expected_x[5] = 9.0; /* (0,2): w_prod/x[0] */

    /* Var 8 (matrix [2,1], col 1): [0, 3] (excludes col 1) */
    expected_x[14] = 0.0;
    expected_x[15] = 3.0; /* (1,2): w_prod/x[1] */

    /* Var 13 (matrix [2,2], col 2): [9, 3] (excludes col 2) */
    expected_x[24] = 9.0; /* (2,0): w_prod/x[0] */
    expected_x[25] = 3.0; /* (2,1): w_prod/x[1] */

    /* Row 3 block (no zeros, scale = 4 * 8 = 32), x = [1,4,2] */
    /* Var 4 (matrix [3,0], col 0): [8, 16] (excludes col 0) */
    expected_x[6] = 8.0;  /* 32/(1*4) */
    expected_x[7] = 16.0; /* 32/(1*2) */

    /* Var 9 (matrix [3,1], col 1): [8, 4] (excludes col 1) */
    expected_x[16] = 8.0; /* 32/(4*1) */
    expected_x[17] = 4.0; /* 32/(4*2) */

    /* Var 14 (matrix [3,2], col 2): [16, 4] (excludes col 2) */
    expected_x[26] = 16.0; /* 32/(2*1) */
    expected_x[27] = 4.0;  /* 32/(2*4) */

    /* Row 4 block (no zeros, scale = 5 * 15 = 75), x = [1,5,3] */
    /* Var 5 (matrix [4,0], col 0): [15, 25] (excludes col 0) */
    expected_x[8] = 15.0; /* 75/(1*5) */
    expected_x[9] = 25.0; /* 75/(1*3) */

    /* Var 10 (matrix [4,1], col 1): [15, 5] (excludes col 1) */
    expected_x[18] = 15.0; /* 75/(5*1) */
    expected_x[19] = 5.0;  /* 75/(5*3) */

    /* Var 15 (matrix [4,2], col 2): [25, 5] (excludes col 2) */
    expected_x[28] = 25.0; /* 75/(3*1) */
    expected_x[29] = 5.0;  /* 75/(3*5) */

    /* Row pointers (monotonically increasing for valid CSR format) */
    int expected_p[17] = {0,  0,  2,  4,  6,  8,  10, 12, 14,
                          16, 18, 20, 22, 24, 26, 28, 30};

    /* Column indices (each row of the matrix interacts with its own columns,
     * excluding diagonal) */
    int expected_i[30];
    for (int var_idx = 0; var_idx < 15; var_idx++)
    {
        int matrix_row = var_idx % 5; /* which row of the 5x3 matrix */
        int col_i = var_idx / 5;      /* which column this variable is in */
        int nnz_start = var_idx * 2;
        /* All columns from matrix_row, excluding col_i */
        int offset = 0;
        for (int j = 0; j < 3; j++)
        {
            if (j != col_i)
            {
                expected_i[nnz_start + offset] = 1 + matrix_row + j * 5;
                offset++;
            }
        }
    }

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 30));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 17));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 30));

    free_expr(p);
    return 0;
}
const char *test_wsum_hess_prod_axis_one_2x2()
{
    /* x is 2x2 variable, global index 0, total 4 vars
     * x = [2, 1, 3, 2] (column-major)
     *     [2, 3]
     *     [1, 2]
     * f = prod_axis_one(x) = [2*3, 1*2] = [6, 2]
     * w = [1, 1]
     *
     * For a 2x2 matrix with var_id=0:
     *   Var 0 (matrix [0,0]): p[0]=0, stores columns [0,1]
     *   Var 1 (matrix [1,0]): p[1]=2, stores columns [0,1]
     *   Var 2 (matrix [0,1]): p[2]=4, stores columns [0,1]
     *   Var 3 (matrix [1,1]): p[3]=6, stores columns [0,1]
     *
     * Row 0 Hessian (no zeros, x=[2,3], scale=1*6=6):
     *   (0,0)=0, (0,1)=1  -> stored at Var 0: [0, 1]
     *   (1,0)=1, (1,1)=0  -> stored at Var 2: [1, 0]
     *
     * Row 1 Hessian (no zeros, x=[1,2], scale=1*2=2):
     *   (0,0)=0, (0,1)=1  -> stored at Var 1: [0, 1]
     *   (1,0)=1, (1,1)=0  -> stored at Var 3: [1, 0]
     */
    double u_vals[4] = {2.0, 1.0, 3.0, 2.0};
    double w_vals[2] = {1.0, 1.0};
    expr *x = new_variable(2, 2, 0, 4);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    /* Expected sparse structure (nnz = 4, each row has 1 nnz) */
    double expected_x[4] = {1.0,  /* Var 0 (excludes col 0) */
                            1.0,  /* Var 1 (excludes col 0) */
                            1.0,  /* Var 2 (excludes col 1) */
                            1.0}; /* Var 3 (excludes col 1) */

    /* Row pointers (each row has 1 nnz) */
    int expected_p[5] = {0, 1, 2, 3, 4};

    /* Column indices (each variable stores columns for its matrix row, excluding
     * diagonal) */
    int expected_i[4] = {2,  /* Var 0 (row 0, col 0): only col 1 */
                         3,  /* Var 1 (row 1, col 0): only col 1 */
                         0,  /* Var 2 (row 0, col 1): only col 0 */
                         1}; /* Var 3 (row 1, col 1): only col 0 */

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 4));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 5));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 4));

    free_expr(p);
    return 0;
}

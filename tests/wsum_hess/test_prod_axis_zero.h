#include <stdio.h>
#include <string.h>

#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_wsum_hess_prod_axis_zero_no_zeros()
{
    /* x is 2x3 variable, global index 1, total 8 vars
     * x = [1, 2, 3, 4, 5, 6] (column-major)
     *     [1, 3, 5]
     *     [2, 4, 6]
     * f = prod_axis_zero(x) = [2, 12, 30]
     * w = [1, 2, 3]
     *
     * Hessian is block diagonal with three 2x2 blocks:
     * Block 0 (col 0): w[0] * f[0] = 1 * 2 = 2
     *   H_00 = [0, 2/(1*2)] = [0, 1]
     *   H_01 = [2/(2*1), 0] = [1, 0]
     * Block 1 (col 1): w[1] * f[1] = 2 * 12 = 24
     *   H_00 = [0, 24/(3*4)] = [0, 2]
     *   H_01 = [24/(4*3), 0] = [2, 0]
     * Block 2 (col 2): w[2] * f[2] = 3 * 30 = 90
     *   H_00 = [0, 90/(5*6)] = [0, 3]
     *   H_01 = [90/(6*5), 0] = [3, 0]
     */
    double u_vals[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    double w_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(2, 3, 1, 8);
    expr *p = new_prod_axis_zero(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    /* Block diagonal structure: 3 blocks of 2x2 = 6 nnz total
     * Each block has 4 entries (2x2 dense)
     */
    double expected_x[12] = {/* Block 0 */
                             0.0, 1.0, 1.0, 0.0,
                             /* Block 1 */
                             0.0, 2.0, 2.0, 0.0,
                             /* Block 2 */
                             0.0, 3.0, 3.0, 0.0};

    /* Row pointers: variable is at global rows 1-6, so:
     * p[0] = 0 (row 0: before variable)
     * p[1-6] = block diagonal entries for the variable
     * p[7-8] = 12 (rows 7-8: after variable)
     */
    int expected_p[9] = {0, 0, 2, 4, 6, 8, 10, 12, 12};

    /* Column indices: block diagonal structure
     * Row 0: cols 1, 2 (global indices for block 0)
     * Row 1: cols 1, 2
     * Row 2: cols 3, 4 (global indices for block 1)
     * Row 3: cols 3, 4
     * Row 4: cols 5, 6 (global indices for block 2)
     * Row 5: cols 5, 6
     */
    int expected_i[12] = {1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 12));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 12));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_zero_mixed_zeros()
{
    /* x is 5x3 variable, global index 1, total 16 vars
     * x = [1, 1, 1, 1, 1, 2, 0, 3, 4, 5, 1, 0, 0, 2, 3] (column-major)
     * Matrix (column-major):
     *     [1, 2, 1]
     *     [1, 0, 0]
     *     [1, 3, 0]
     *     [1, 4, 2]
     *     [1, 5, 3]
     *
     * f = prod_axis_zero(x) = [1, 120, 6]
     * Column 0: 5 nonzeros, prod = 1*1*1*1*1 = 1
     * Column 1: 1 zero at row 1, prod_nonzero = 2*3*4*5 = 120
     * Column 2: 2 zeros at rows 1,2, prod_nonzero = 1*2*3 = 6
     *
     * w = [1, 2, 3]
     *
     * Block 0 (5x5): no zeros, diagonal=0, off-diagonal=w[0]*f[0]/(x[i]*x[j]) =
     * 1/(1*1) = 1
     * Block 1 (5x5): 1 zero at row 1, nonzeros only in row/col 1
     *   (1,0): 2*120/5 = 48, (1,2): 2*120/3 = 80, (1,3): 2*120/4 = 60, (1,4):
     * 2*120/5 = 48 Block 2 (5x5): 2 zeros at rows 1,2, only (1,3) and (2,3) are
     * nonzero (1,3): 3*6/2 = 9, (2,3): 3*6/3 = 6
     */
    double u_vals[16] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0,
                         3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 2.0, 3.0};
    /* This maps to: x[var_id+0..4]=column 0, x[var_id+5..9]=column 1,
     * x[var_id+10..14]=column 2 Column 0: [1, 1, 1, 1, 1] - all nonzero Column 1:
     * [2, 0, 3, 4, 5] - one zero at index 1 Column 2: [1, 0, 0, 2, 3] - two zeros at
     * indices 1, 2
     */
    double w_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(5, 3, 1, 16);
    expr *p = new_prod_axis_zero(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    /* Block 0: 5x5 all off-diagonal = 1.0, total = 25 entries (indices 0-24)
     * Block 1: 5x5 with 1 zero, total = 25 entries (indices 25-49)
     * Block 2: 5x5 with 2 zeros, total = 25 entries (indices 50-74)
     * Total nnz = 75
     */
    double expected_x[75];
    memset(expected_x, 0, sizeof(expected_x));

    /* Block 0 (indices 0-24): all off-diagonal = 1.0 */
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (i != j)
            {
                expected_x[i * 5 + j] = 1.0;
            }
        }
    }

    /* Block 1 (indices 25-49): 1 zero at row 1 (index p=1), w[1]*prod_nonzero[1] =
     * 2*120 = 240 Column 1 is x[5..9] = [2, 0, 3, 4, 5] (rows 0-4) Only row p=1 and
     * column p=1 have nonzeros H[i,p] = H[p,i] = w_prod / x[i] for i != p H[0,1] =
     * H[1,0] = 240/2 = 120 H[2,1] = H[1,2] = 240/3 = 80 H[3,1] = H[1,3] = 240/4 = 60
     * H[4,1] = H[1,4] = 240/5 = 48
     */
    /* Row 0 of block 1: only (0,1) is nonzero */
    expected_x[25 + 0 * 5 + 1] = 120.0;
    /* Row 1 of block 1: (1,0), (1,2), (1,3), (1,4) are nonzero */
    expected_x[25 + 1 * 5 + 0] = 120.0;
    expected_x[25 + 1 * 5 + 2] = 80.0;
    expected_x[25 + 1 * 5 + 3] = 60.0;
    expected_x[25 + 1 * 5 + 4] = 48.0;
    /* Row 2 of block 1: only (2,1) is nonzero */
    expected_x[25 + 2 * 5 + 1] = 80.0;
    /* Row 3 of block 1: only (3,1) is nonzero */
    expected_x[25 + 3 * 5 + 1] = 60.0;
    /* Row 4 of block 1: only (4,1) is nonzero */
    expected_x[25 + 4 * 5 + 1] = 48.0;

    /* Block 2 (indices 50-74): 2 zeros at rows p=1, q=2, w[2]*prod_nonzero[2] = 3*6
     * = 18 Column 2 is x[10..14] = [1, 0, 0, 2, 3] (rows 0-4) With exactly 2 zeros,
     * only H[p,q] and H[q,p] are nonzero H[1,2] = H[2,1] = 18
     */
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) expected_x[50 + i * 5 + j] = 0.0;
    expected_x[50 + 1 * 5 + 2] = 18.0; /* H[1,2] */
    expected_x[50 + 2 * 5 + 1] = 18.0; /* H[2,1] */

    int expected_p[17] = {0,  0,  5,  10, 15, 20, 25, 30, 35,
                          40, 45, 50, 55, 60, 65, 70, 75};

    /* Column indices: block diagonal structure
     * Block 0 (rows 1-5): each row has cols [1, 2, 3, 4, 5]
     * Block 1 (rows 6-10): each row has cols [6, 7, 8, 9, 10]
     * Block 2 (rows 11-15): each row has cols [11, 12, 13, 14, 15]
     */
    int expected_i[75];
    for (int block = 0; block < 3; block++)
    {
        int offset = block * 25;
        int col_start = block * 5 + 1;
        for (int row = 0; row < 5; row++)
        {
            for (int col = 0; col < 5; col++)
            {
                expected_i[offset + row * 5 + col] = col_start + col;
            }
        }
    }

    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 17));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 75));
    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 75));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_zero_one_zero()
{
    /* Test with a column that has exactly 1 zero
     * x is 2x2 variable, global index 1, total 5 vars
     * x = [1.0, 1.0, 2.0, 0.0] (column-major)
     * Matrix (column-major):
     *     [1, 2]
     *     [1, 0]
     *
     * f = prod_axis_zero(x) = [1, 0]
     * Column 0: no zeros, prod = 1
     * Column 1: 1 zero at row 1, prod_nonzero = 2
     * w = [1, 2]
     */
    double u_vals[5] = {0.0, 1.0, 1.0, 2.0, 0.0};
    double w_vals[2] = {1.0, 2.0};
    expr *x = new_variable(2, 2, 1, 5);
    expr *p = new_prod_axis_zero(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    /* Block 0 (no zeros): w[0]*f[0] = 1
     *   (0,1) = 1/(1*1) = 1, (1,0) = 1
     * Block 1 (1 zero at row 1): w[1]*prod_nonzero[1] = 2*2 = 4
     *   (1,0) = 4/2 = 2 (symmetric)
     */
    double expected_x[8];
    memset(expected_x, 0, sizeof(expected_x));

    /* Block 0 */
    expected_x[0 * 2 + 1] = 1.0; /* (0,1) */
    expected_x[1 * 2 + 0] = 1.0; /* (1,0) */

    /* Block 1: one zero at row 1 */
    expected_x[4 + 1 * 2 + 0] = 2.0; /* (1,0) */
    expected_x[4 + 0 * 2 + 1] = 2.0; /* (0,1) symmetric */

    /* Row pointers: variable is at global rows 1-2, so:
     * p[0] = 0 (row 0: before variable)
     * p[1-2] = block 0 (rows 1-2, each has 2 entries)
     * p[3-4] = block 1 (rows 3-4, each has 2 entries)
     * p[5] = 8 (after variable)
     */
    int expected_p[6] = {0, 0, 2, 4, 6, 8};

    /* Column indices: block diagonal structure
     * Block 0: cols 1, 2 (global indices for block 0)
     * Block 1: cols 3, 4 (global indices for block 1)
     */
    int expected_i[8] = {1, 2, 1, 2, 3, 4, 3, 4};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 8));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 6));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 8));

    free_expr(p);
    return 0;
}

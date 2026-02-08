#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSR_Matrix.h"
#include "utils/int_double_pair.h"

const char *test_diag_csr_mult()
{
    /* Create a 3x3 CSR matrix A:
     * [1.0  2.0  0.0]
     * [0.0  3.0  4.0]
     * [5.0  0.0  6.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 3, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 1, 1, 2, 0, 2};
    int Ap[4] = {0, 2, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));
    double d[3] = {2.0, 3.0, 0.5};

    /* Result C should be diag(d) * A:
     * [2.0  4.0  0.0]
     * [0.0  9.0  12.0]
     * [2.5  0.0  3.0]
     */
    CSR_Matrix *C = new_csr_matrix(3, 3, 6);
    diag_csr_mult(d, A, C);

    double Ax_correct[6] = {2.0, 4.0, 9.0, 12.0, 2.5, 3.0};
    int Ai_correct[6] = {0, 1, 1, 2, 0, 2};
    int Ap_correct[4] = {0, 2, 4, 6};

    mu_assert("vals incorrect", cmp_double_array(C->x, Ax_correct, 6));
    mu_assert("cols incorrect", cmp_int_array(C->i, Ai_correct, 6));
    mu_assert("rows incorrect", cmp_int_array(C->p, Ap_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(C);

    return 0;
}

/*
[1  0  2]   [0 1 0]     [1 1 2]
[0  3  0] + [2 0 3] =   [2 3 3]
[4  0  5]   [0 6 0]     [4 6 5]
*/
const char *test_csr_sum()
{
    CSR_Matrix *A = new_csr_matrix(3, 3, 5);
    double Ax[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int Ai[5] = {0, 2, 1, 0, 2};
    int Ap[4] = {0, 2, 3, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    CSR_Matrix *B = new_csr_matrix(3, 3, 4);
    double Bx[4] = {1.0, 2.0, 3.0, 6.0};
    int Bi[4] = {1, 0, 2, 1};
    int Bp[4] = {0, 1, 3, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    CSR_Matrix *C = new_csr_matrix(3, 3, 9);
    sum_csr_matrices(A, B, C);

    double Cx_correct[9] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 6.0, 5.0};
    int Ci_correct[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int Cp_correct[4] = {0, 3, 6, 9};

    mu_assert("C nnz incorrect", C->nnz == 9);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 9));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 9));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_csr_matrix(C);

    return 0;
}

/*
[1  0  2]   [0 1 0]     [1 1 2]
[0  0  3] + [2 0 3] =   [2 0 6]
[4  0  5]   [0 6 0]     [4 6 5]
*/
const char *test_csr_sum2()
{
    CSR_Matrix *A = new_csr_matrix(3, 3, 5);
    double Ax[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int Ai[5] = {0, 2, 2, 0, 2};
    int Ap[4] = {0, 2, 3, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    CSR_Matrix *B = new_csr_matrix(3, 3, 4);
    double Bx[4] = {1.0, 2.0, 3.0, 6.0};
    int Bi[4] = {1, 0, 2, 1};
    int Bp[4] = {0, 1, 3, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    CSR_Matrix *C = new_csr_matrix(3, 3, 8);
    sum_csr_matrices(A, B, C);

    double Cx_correct[8] = {1, 1, 2, 2, 6, 4, 6, 5};
    int Ci_correct[8] = {0, 1, 2, 0, 2, 0, 1, 2};
    int Cp_correct[4] = {0, 3, 5, 8};

    mu_assert("C nnz incorrect", C->nnz == 8);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 8));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 8));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_csr_matrix(C);

    return 0;
}

const char *test_transpose()
{
    CSR_Matrix *A = new_csr_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 1, 0, 1};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    int iwork[5];
    CSR_Matrix *AT = transpose(A, iwork);
    double ATx_correct[5] = {1.0, 2.0, 3.0, 4.0, 1.0};
    int ATi_correct[5] = {0, 2, 1, 3, 0};
    int ATp_correct[6] = {0, 2, 4, 4, 4, 5};
    mu_assert("AT vals incorrect", cmp_double_array(AT->x, ATx_correct, 5));
    mu_assert("AT cols incorrect", cmp_int_array(AT->i, ATi_correct, 5));
    mu_assert("AT rows incorrect", cmp_int_array(AT->p, ATp_correct, 6));

    free_csr_matrix(A);
    free_csr_matrix(AT);

    return 0;
}

/* C = z^T A where

z = (1, 2, 3, 4) and
A = [1 0 0 0 1
     0 3 0 0 0
     2 0 0 0 0
     0 4 0 0 0]
*/
const char *test_csr_vecmat_values_sparse()
{
    CSR_Matrix *A = new_csr_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 1, 0, 1};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    double z[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *C = new_csr_matrix(1, 3, 3);
    double Cx[3] = {0.0, 0.0, 0.0};
    int Ci[3] = {0, 1, 4};
    int Cp[2] = {0, 3};
    memcpy(C->x, Cx, 3 * sizeof(double));
    memcpy(C->i, Ci, 3 * sizeof(int));
    memcpy(C->p, Cp, 2 * sizeof(int));

    int iwork[5];

    CSR_Matrix *AT = transpose(A, iwork);

    csr_matvec_fill_values(AT, z, C);

    double Cx_correct[3] = {7.0, 22.0, 1.0};

    mu_assert("C nnz incorrect", C->nnz == 3);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 3));

    free_csr_matrix(A);
    free_csr_matrix(AT);
    free_csr_matrix(C);

    return 0;
}
const char *test_sum_all_rows_csr()
{
    /* Create a 3x4 CSR matrix A:
     * [1.0  2.0  0.0  0.0]
     * [0.0  3.0  4.0  0.0]
     * [5.0  0.0  6.0  7.0]
     *
     * Sum all rows should give:
     * [6.0  5.0  10.0  7.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 4, 7);
    double Ax[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    int Ai[7] = {0, 1, 1, 2, 0, 2, 3};
    int Ap[4] = {0, 2, 4, 7};
    memcpy(A->x, Ax, 7 * sizeof(double));
    memcpy(A->i, Ai, 7 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));
    CSR_Matrix *C = new_csr_matrix(1, 4, 4);
    int_double_pair *pairs = new_int_double_pair_array(7);
    sum_all_rows_csr(A, C, pairs);
    double Cx_correct[4] = {6.0, 5.0, 10.0, 7.0};
    int Ci_correct[4] = {0, 1, 2, 3};
    int Cp_correct[2] = {0, 4};

    mu_assert("C nnz incorrect", C->nnz == 4);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 4));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 4));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 2));

    free_csr_matrix(A);
    free_csr_matrix(C);
    free_int_double_pair_array(pairs);

    return 0;
}
const char *test_sum_block_of_rows_csr()
{
    /* Create a 9x4 CSR matrix A and sum blocks of size 3
     * Block 0 (rows 0-2):
     * [1.0  2.0  0.0  0.0]
     * [0.0  3.0  1.0  0.0]
     * [0.0  0.0  4.0  5.0]
     * Sum: [1.0  5.0  5.0  5.0]
     *
     * Block 1 (rows 3-5):
     * [2.0  0.0  0.0  1.0]
     * [0.0  1.0  2.0  0.0]
     * [3.0  0.0  0.0  0.0]
     * Sum: [5.0  1.0  2.0  1.0]
     *
     * Block 2 (rows 6-8):
     * [0.0  4.0  0.0  0.0]
     * [1.0  0.0  3.0  0.0]
     * [0.0  2.0  0.0  6.0]
     * Sum: [1.0  6.0  3.0  6.0]
     *
     * Result C should be 3x4 matrix with the sums above
     */
    CSR_Matrix *A = new_csr_matrix(9, 4, 18);

    double Ax[18] = {1.0, 2.0,  /* row 0 */
                     3.0, 1.0,  /* row 1 */
                     4.0, 5.0,  /* row 2 */
                     2.0, 1.0,  /* row 3 */
                     1.0, 2.0,  /* row 4 */
                     3.0,       /* row 5 */
                     4.0,       /* row 6 */
                     1.0, 3.0,  /* row 7 */
                     2.0, 6.0}; /* row 8 */

    int Ai[18] = {0, 1,  /* row 0 */
                  1, 2,  /* row 1 */
                  2, 3,  /* row 2 */
                  0, 3,  /* row 3 */
                  1, 2,  /* row 4 */
                  0,     /* row 5 */
                  1,     /* row 6 */
                  0, 2,  /* row 7 */
                  1, 3}; /* row 8 */

    int Ap[10] = {0, 2, 4, 6, 8, 10, 11, 12, 14, 16};

    memcpy(A->x, Ax, 18 * sizeof(double));
    memcpy(A->i, Ai, 18 * sizeof(int));
    memcpy(A->p, Ap, 10 * sizeof(int));

    /* Allocate C for 3 blocks and enough space for all nonzeros */
    CSR_Matrix *C = new_csr_matrix(3, 4, 12);
    int_double_pair *pairs = new_int_double_pair_array(18);

    sum_block_of_rows_csr(A, C, pairs, 3);

    /* Expected results for 3 blocks */
    double Cx_correct[12] = {1.0, 5.0, 5.0, 5.0,  /* block 0 sum */
                             5.0, 1.0, 2.0, 1.0,  /* block 1 sum */
                             1.0, 6.0, 3.0, 6.0}; /* block 2 sum */

    int Ci_correct[12] = {0, 1, 2, 3,  /* block 0 columns */
                          0, 1, 2, 3,  /* block 1 columns */
                          0, 1, 2, 3}; /* block 2 columns */

    int Cp_correct[4] = {0, 4, 8, 12};

    mu_assert("C nnz incorrect", C->nnz == 12);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 12));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 12));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(C);
    free_int_double_pair_array(pairs);

    return 0;
}
const char *test_sum_evenly_spaced_rows_csr()
{
    /* Create a 9x4 CSR matrix A (same as test_sum_block_of_rows_csr) and sum evenly
     * spaced rows With row_spacing=3:


    A = 9x4 CSR matrix:
            1 2 0 0
            0 3 1 0
            0 0 4 5
            2 0 0 1
            0 1 2 0
            3 0 0 0
            0 4 0 0
            1 0 3 0
            0 2 0 6

    Result C should be 3x4 matrix:
            row 0: sum of rows 0, 3, 6 = [3 6 0 1]
            row 1: sum of rows 1, 4, 7 = [1 4 6 0]
            row 2: sum of rows 2, 5, 8 = [3 2 4 11]
    */
    CSR_Matrix *A = new_csr_matrix(9, 4, 18);

    double Ax[18] = {1.0, 2.0,  /* row 0 */
                     3.0, 1.0,  /* row 1 */
                     4.0, 5.0,  /* row 2 */
                     2.0, 1.0,  /* row 3 */
                     1.0, 2.0,  /* row 4 */
                     3.0,       /* row 5 */
                     4.0,       /* row 6 */
                     1.0, 3.0,  /* row 7 */
                     2.0, 6.0}; /* row 8 */

    int Ai[18] = {0, 1,  /* row 0 */
                  1, 2,  /* row 1 */
                  2, 3,  /* row 2 */
                  0, 3,  /* row 3 */
                  1, 2,  /* row 4 */
                  0,     /* row 5 */
                  1,     /* row 6 */
                  0, 2,  /* row 7 */
                  1, 3}; /* row 8 */

    int Ap[10] = {0, 2, 4, 6, 8, 10, 11, 12, 14, 16};

    memcpy(A->x, Ax, 18 * sizeof(double));
    memcpy(A->i, Ai, 18 * sizeof(int));
    memcpy(A->p, Ap, 10 * sizeof(int));

    /* Allocate C for 3 rows (row_spacing=3) and enough space for all nonzeros */
    CSR_Matrix *C = new_csr_matrix(3, 4, 10);
    int_double_pair *pairs = new_int_double_pair_array(18);

    sum_evenly_spaced_rows_csr(A, C, pairs, 3);

    /* Expected results for evenly spaced rows */
    double Cx_correct[10] = {3.0, 6.0, 1.0,        /* output row 0 */
                             1.0, 4.0, 6.0,        /* output row 1 */
                             3.0, 2.0, 4.0, 11.0}; /* output row 2 */

    int Ci_correct[10] = {0, 1, 3,     /* output row 0 columns */
                          0, 1, 2,     /* output row 1 columns */
                          0, 1, 2, 3}; /* output row 2 columns */

    int Cp_correct[4] = {0, 3, 6, 10};

    mu_assert("C nnz incorrect", C->nnz == 10);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 10));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 10));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(C);
    free_int_double_pair_array(pairs);

    return 0;
}
const char *test_AT_alloc_and_fill()
{
    /* Create a 3x4 CSR matrix A:
     * [1.0  0.0  2.0  0.0]
     * [0.0  3.0  0.0  4.0]
     * [5.0  0.0  6.0  0.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 4, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 2, 1, 3, 0, 2};
    int Ap[4] = {0, 2, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* Allocate A^T (should be 4x3) */
    int *iwork = (int *) malloc(A->n * sizeof(int));
    CSR_Matrix *AT = AT_alloc(A, iwork);

    /* Fill values of A^T */
    AT_fill_values(A, AT, iwork);

    /* Expected A^T:
     * [1.0  0.0  5.0]
     * [0.0  3.0  0.0]
     * [2.0  0.0  6.0]
     * [0.0  4.0  0.0]
     */
    double ATx_correct[6] = {1.0, 5.0, 3.0, 2.0, 6.0, 4.0};
    int ATi_correct[6] = {0, 2, 1, 0, 2, 1};
    int ATp_correct[5] = {0, 2, 3, 5, 6};

    mu_assert("AT dimensions incorrect", AT->m == 4 && AT->n == 3);
    mu_assert("AT nnz incorrect", AT->nnz == 6);
    mu_assert("AT vals incorrect", cmp_double_array(AT->x, ATx_correct, 6));
    mu_assert("AT cols incorrect", cmp_int_array(AT->i, ATi_correct, 6));
    mu_assert("AT rows incorrect", cmp_int_array(AT->p, ATp_correct, 5));

    free_csr_matrix(A);
    free_csr_matrix(AT);
    free(iwork);

    return 0;
}

const char *test_kron_identity_csr()
{
    /* Test A kron I_p where:
     * A is 2x3:
     * [1  0  2]
     * [3  0  4]
     *
     * p = 2, so result should be 4x6 (2*2 rows, 3*2 cols)
     * A kron I_2 =
     * [1  0 | 0  0 | 2  0]
     * [0  1 | 0  0 | 0  2]
     * [-----|-----|-----]
     * [3  0 | 0  0 | 4  0]
     * [0  3 | 0  0 | 0  4]
     */
    CSR_Matrix *A = new_csr_matrix(2, 3, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 2, 0, 2};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    CSR_Matrix *result = kron_identity_csr(A, 2);

    /* Expected: 4x6 with 8 nonzeros
     * Row 0: [1, 0, 0, 0, 2, 0] -> cols {0, 4}, vals {1, 2}
     * Row 1: [0, 1, 0, 0, 0, 2] -> cols {1, 5}, vals {1, 2}
     * Row 2: [3, 0, 0, 0, 4, 0] -> cols {0, 4}, vals {3, 4}
     * Row 3: [0, 3, 0, 0, 0, 4] -> cols {1, 5}, vals {3, 4}
     */
    double expected_x[8] = {1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0};
    int expected_i[8] = {0, 4, 1, 5, 0, 4, 1, 5};
    int expected_p[5] = {0, 2, 4, 6, 8};

    mu_assert("dimensions incorrect", result->m == 4 && result->n == 6);
    mu_assert("nnz incorrect", result->nnz == 8);
    mu_assert("vals incorrect", cmp_double_array(result->x, expected_x, 8));
    mu_assert("cols incorrect", cmp_int_array(result->i, expected_i, 8));
    mu_assert("rows incorrect", cmp_int_array(result->p, expected_p, 5));

    free_csr_matrix(A);
    free_csr_matrix(result);

    return 0;
}

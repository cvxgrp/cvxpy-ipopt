#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"

const char *test_csr_to_csc1()
{
    CSR_Matrix *A = new_csr_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 1, 0, 1};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    CSC_Matrix *C = csr_to_csc(A);

    double Cx_correct[5] = {1.0, 2.0, 3.0, 4.0, 1.0};
    int Ci_correct[5] = {0, 2, 1, 3, 0};
    int Cp_correct[6] = {0, 2, 4, 4, 4, 5};

    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 5));
    mu_assert("C rows incorrect", cmp_int_array(C->i, Ci_correct, 5));
    mu_assert("C cols incorrect", cmp_int_array(C->p, Cp_correct, 6));

    free_csr_matrix(A);
    free_csc_matrix(C);

    return 0;
}

const char *test_csr_to_csc2()
{
    CSR_Matrix *A = new_csr_matrix(20, 30, 120);
    double Ax[120] = {9, 6, 5, 9, 7, 3, 8, 2, 6, 1, 3, 9, 2, 8, 9, 1, 4, 9, 2, 1,
                      3, 4, 2, 8, 6, 2, 9, 7, 3, 8, 3, 7, 9, 2, 2, 2, 5, 5, 3, 5,
                      1, 6, 7, 2, 7, 3, 3, 7, 3, 5, 4, 7, 7, 3, 6, 3, 6, 1, 8, 8,
                      3, 2, 2, 3, 4, 5, 5, 5, 8, 3, 5, 3, 7, 5, 1, 4, 9, 6, 6, 7,
                      4, 6, 8, 2, 7, 3, 5, 3, 3, 4, 7, 3, 6, 4, 2, 1, 1, 5, 5, 8,
                      1, 9, 5, 2, 3, 8, 5, 8, 4, 5, 5, 6, 9, 6, 4, 4, 1, 8, 9, 8};
    int Ai[120] = {1,  2,  3,  19, 21, 22, 9,  10, 19, 20, 25, 0,  6,  8,  9,
                   12, 15, 19, 20, 21, 26, 2,  5,  6,  8,  12, 14, 16, 19, 27,
                   8,  11, 13, 15, 25, 26, 27, 10, 12, 19, 22, 23, 24, 25, 28,
                   1,  11, 12, 15, 18, 24, 13, 22, 2,  5,  6,  9,  18, 24, 3,
                   6,  8,  22, 20, 27, 7,  9,  17, 26, 29, 0,  1,  11, 13, 15,
                   16, 18, 23, 24, 4,  5,  8,  9,  16, 20, 23, 4,  6,  14, 15,
                   24, 8,  9,  11, 12, 20, 22, 29, 2,  5,  12, 14, 15, 19, 21,
                   10, 19, 27, 1,  5,  6,  9,  11, 15, 21, 26, 3,  15, 26, 27};
    int Ap[21] = {0,  6,  11, 21, 30, 37, 45,  51,  53,  59, 63,
                  65, 70, 79, 86, 91, 98, 105, 108, 116, 120};
    memcpy(A->x, Ax, 120 * sizeof(double));
    memcpy(A->i, Ai, 120 * sizeof(int));
    memcpy(A->p, Ap, 21 * sizeof(int));

    CSC_Matrix *C = csr_to_csc(A);

    double Cx_correct[120] = {
        9, 5, 9, 3, 3, 4, 6, 4, 3, 5, 5, 8, 1, 7, 5, 2, 6, 4, 8, 5, 2, 8, 3, 3,
        3, 5, 5, 8, 6, 3, 2, 6, 3, 8, 9, 6, 5, 8, 6, 6, 2, 5, 8, 7, 3, 7, 4, 9,
        1, 2, 3, 7, 2, 1, 9, 7, 5, 9, 3, 9, 4, 2, 3, 1, 4, 5, 6, 8, 7, 4, 2, 5,
        5, 1, 9, 9, 6, 9, 3, 5, 2, 5, 1, 2, 3, 7, 1, 7, 1, 3, 4, 3, 1, 7, 2, 1,
        6, 6, 3, 7, 4, 8, 6, 7, 3, 2, 2, 3, 2, 8, 4, 9, 8, 5, 4, 8, 8, 7, 3, 5};
    int Ci_correct[120] = {
        2,  12, 0,  6,  12, 18, 0,  3,  8,  16, 0,  9,  19, 13, 14, 3,  8,  13,
        16, 18, 2,  3,  8,  9,  14, 18, 11, 2,  3,  4,  9,  13, 15, 1,  2,  8,
        11, 13, 15, 18, 1,  5,  17, 4,  6,  12, 15, 18, 2,  3,  5,  6,  15, 16,
        4,  7,  12, 3,  14, 16, 2,  4,  6,  12, 14, 16, 18, 19, 3,  12, 13, 11,
        6,  8,  12, 0,  1,  2,  3,  5,  16, 17, 1,  2,  10, 13, 15, 0,  2,  16,
        18, 0,  5,  7,  9,  15, 5,  12, 13, 5,  6,  8,  12, 14, 1,  4,  5,  2,
        4,  11, 18, 19, 3,  4,  10, 17, 19, 5,  11, 15};
    int Cp_correct[31] = {0,  2,  6,  10,  13,  15,  20,  26,  27, 33, 40,
                          43, 48, 54, 57,  60,  68,  71,  72,  75, 82, 87,
                          91, 96, 99, 104, 107, 112, 117, 118, 120};

    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 120));
    mu_assert("C rows incorrect", cmp_int_array(C->i, Ci_correct, 120));
    mu_assert("C cols incorrect", cmp_int_array(C->p, Cp_correct, 31));

    free_csr_matrix(A);
    free_csc_matrix(C);

    return 0;
}

/* Test ATA_alloc with a simple 3x3 example
 * A is 4x3 (4 rows, 3 columns):
 * [x  0  x]
 * [0  x  0]
 * [0  0  x]
 * [0  x  0]
 *
 * A^T A is 3x3:
 * [x  0  x]
 * [0  x  0]
 * [x  0  x]
 */
const char *test_ATA_alloc_simple()
{
    CSC_Matrix *A = new_csc_matrix(4, 3, 6);
    int Ap[4] = {0, 2, 3, 6};
    int Ai[5] = {0, 2, 1, 2, 1};
    memcpy(A->p, Ap, 4 * sizeof(int));
    memcpy(A->i, Ai, 5 * sizeof(int));

    /* Compute C = A^T A */
    CSR_Matrix *C = ATA_alloc(A);
    int expected_p[4] = {0, 2, 3, 5};
    int expected_i[5] = {0, 2, 1, 0, 2};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 4));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 5);

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}

/* Test ATA_alloc with a sparse 3x4 matrix with no overlaps on some pairs
 * A is 3x4:
 * [1  0  0  2]
 * [0  1  0  0]
 * [0  0  1  0]
 *
 * A^T A is 4x4:
 * [1  0  0  2]
 * [0  1  0  0]
 * [0  0  1  0]
 * [2  0  0  4]
 *
 */
const char *test_ATA_alloc_diagonal_like()
{
    /* Create A in CSC format (3 rows, 4 cols, 4 nonzeros) */
    CSC_Matrix *A = new_csc_matrix(3, 4, 4);
    int Ap[5] = {0, 1, 2, 3, 4};
    int Ai[4] = {0, 1, 2, 0};
    memcpy(A->p, Ap, 5 * sizeof(int));
    memcpy(A->i, Ai, 4 * sizeof(int));
    CSR_Matrix *C = ATA_alloc(A);

    int expected_p[5] = {0, 2, 3, 4, 6};
    int expected_i[6] = {0, 3, 1, 2, 0, 3};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 5));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 6);

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}

const char *test_ATA_alloc_random()
{
    /* Create A in CSC format  */
    CSC_Matrix *A = new_csc_matrix(10, 15, 15);
    int Ap[16] = {0, 1, 1, 1, 1, 4, 5, 6, 7, 8, 9, 11, 11, 11, 13, 15};
    int Ai[15] = {5, 0, 6, 9, 0, 5, 1, 3, 6, 0, 6, 3, 6, 6, 8};
    double Ax[15] = {7, 4, 8, 5, 7, 3, 7, 8, 5, 4, 8, 8, 3, 6, 5};
    memcpy(A->p, Ap, 16 * sizeof(int));
    memcpy(A->i, Ai, 15 * sizeof(int));
    memcpy(A->x, Ax, 15 * sizeof(double));
    CSR_Matrix *C = ATA_alloc(A);

    int expected_p[16] = {0, 2, 2, 2, 2, 8, 11, 13, 14, 16, 21, 27, 27, 27, 33, 38};
    int expected_i[38] = {0,  6, 4,  5, 9,  10, 13, 14, 4, 5,  10, 0,  6,
                          7,  8, 13, 4, 9,  10, 13, 14, 4, 5,  9,  10, 13,
                          14, 4, 8,  9, 10, 13, 14, 4,  9, 10, 13, 14};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 16));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 38);

    double d[10] = {2, 8, 6, 2, 5, 1, 6, 9, 1, 3};

    ATDA_fill_values(A, d, C);

    double Cx_correct[38] = {
        49.,  21.,  491., 56.,  240., 416., 144., 288., 56.,  98.,  56.,  21.,  9.,
        392., 128., 128., 240., 150., 240., 90.,  180., 416., 56.,  240., 416., 144.,
        288., 144., 128., 90.,  144., 182., 108., 288., 180., 288., 108., 241.};
    mu_assert("x incorrect", cmp_double_array(C->x, Cx_correct, C->nnz));

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}

const char *test_ATA_alloc_random2()
{
    /* Create A in CSC format  */
    int m = 15;
    int n = 10;
    CSC_Matrix *A = new_csc_matrix(m, n, 15);
    int Ap[11] = {0, 2, 4, 6, 6, 9, 12, 12, 14, 14, 15};
    int Ai[15] = {9, 12, 3, 4, 1, 6, 4, 8, 13, 1, 3, 7, 5, 13, 6};
    double Ax[15] = {0.99, 0.9,  0.51, 0.64, 0.39, 0.29, 0.26, 0.91,
                     0.35, 0.18, 0.33, 0.73, 0.97, 0.86, 1.03};
    memcpy(A->p, Ap, 11 * sizeof(int));
    memcpy(A->i, Ai, 15 * sizeof(int));
    memcpy(A->x, Ax, 15 * sizeof(double));
    CSR_Matrix *C = ATA_alloc(A);

    int expected_p[11] = {0, 1, 4, 7, 7, 10, 13, 13, 15, 15, 17};
    int expected_i[17] = {0, 1, 4, 5, 2, 5, 9, 1, 4, 7, 1, 2, 5, 4, 7, 2, 9};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 11));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 17);
    double d[15] = {-0.6,  -0.23, -0.29, -1.36, 0.4,   0.36, 0.11, -0.13,
                    -1.32, -0.32, -0.24, -0.7,  -0.06, 0.5,  1.99};

    ATDA_fill_values(A, d, C);

    double Cx_correct[17] = {-0.362232, -0.189896, 0.06656,   -0.228888, -0.025732,
                             -0.016146, 0.032857,  0.06656,   -1.004802, 0.1505,
                             -0.228888, -0.016146, -0.224833, 0.1505,    0.708524,
                             0.032857,  0.116699};
    mu_assert("x incorrect", cmp_double_array(C->x, Cx_correct, C->nnz));

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}
const char *test_BTA_alloc_and_BTDA_fill()
{
    /* Create A: 4x3 CSC matrix
     * [1.0  0.0  2.0]
     * [0.0  3.0  0.0]
     * [4.0  0.0  5.0]
     * [0.0  6.0  0.0]
     */
    int m = 4;
    int n = 3;
    CSC_Matrix *A = new_csc_matrix(m, n, 6);
    int Ap_A[4] = {0, 2, 4, 6};
    int Ai_A[6] = {0, 2, 1, 3, 0, 2};
    double Ax_A[6] = {1.0, 4.0, 3.0, 6.0, 2.0, 5.0};
    memcpy(A->p, Ap_A, 4 * sizeof(int));
    memcpy(A->i, Ai_A, 6 * sizeof(int));
    memcpy(A->x, Ax_A, 6 * sizeof(double));

    /* Create B: 4x2 CSC matrix
     * [1.0  0.0]
     * [0.0  2.0]
     * [3.0  0.0]
     * [0.0  4.0]
     */
    int p = 2;
    CSC_Matrix *B = new_csc_matrix(m, p, 4);
    int Bp[3] = {0, 2, 4};
    int Bi[4] = {0, 2, 1, 3};
    double Bx[4] = {1.0, 3.0, 2.0, 4.0};
    memcpy(B->p, Bp, 3 * sizeof(int));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->x, Bx, 4 * sizeof(double));

    /* Allocate C = B^T A (should be 2x3) */
    CSR_Matrix *C = BTA_alloc(A, B);

    /* Sparsity pattern check before filling values */
    int expected_p[3] = {0, 2, 3};
    int expected_i[3] = {0, 2, 1};
    mu_assert("C dimensions incorrect", C->m == 2 && C->n == 3);
    mu_assert("C nnz incorrect", C->nnz == 3);
    mu_assert("C->p incorrect", cmp_int_array(C->p, expected_p, 3));
    mu_assert("C->i incorrect", cmp_int_array(C->i, expected_i, 3));

    /* Fill values with diagonal weights d */
    double d[4] = {1.0, 2.0, 3.0, 4.0};
    BTDA_fill_values(A, B, d, C);

    double expected_x[3] = {37.0, 47.0, 108.0};
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x, 3));

    free_csr_matrix(C);
    free_csc_matrix(A);
    free_csc_matrix(B);

    return 0;
}

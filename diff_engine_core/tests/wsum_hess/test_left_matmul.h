#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_left_matmul()
{
    /* Test weighted sum of Hessian of A @ log(x) where:
     * x is 3x1 variable at x = [1, 2, 3]
     * A is 4x3 sparse matrix [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     * Output: A @ log(x) is 4x1
     * Weights w = [1, 2, 3, 4]
     *
     * For log(x_i), the Hessian is diagonal with h_ii = -1/x_i^2
     * The output f = A @ log(x) has:
     * f[0] = log(x[0]) + 2*log(x[2])
     * f[1] = 3*log(x[0]) + 4*log(x[2])
     * f[2] = 5*log(x[0]) + 6*log(x[2])
     * f[3] = 7*log(x[0])
     *
     * Weighted sum of Hessian: sum_k w[k] * d²f[k]/dx²
     * For each variable x[i], we need to find which outputs f[k] depend on it.
     *
     * x[0] affects f[0], f[1], f[2], f[3] with coefficients 1, 3, 5, 7
     * x[1] doesn't affect any output (column 1 of A is all zeros)
     * x[2] affects f[0], f[1], f[2] with coefficients 2, 4, 6
     *
     * d²f[k]/dx[i]² = A[k,i] * (-1/x[i]²)
     *
     * wsum_hess[i,i] = sum_k w[k] * A[k,i] * (-1/x[i]²)
     *                = (-1/x[i]²) * sum_k w[k] * A[k,i]
     *                = (-1/x[i]²) * (A^T @ w)[i]
     *
     * A^T @ w:
     * [1, 3, 5, 7] @ [1, 2, 3, 4] = 1*1 + 3*2 + 5*3 + 7*4 = 1 + 6 + 15 + 28 = 50
     * [0, 0, 0, 0] @ [1, 2, 3, 4] = 0
     * [2, 4, 6, 0] @ [1, 2, 3, 4] = 2*1 + 4*2 + 6*3 = 2 + 8 + 18 = 28
     *
     * wsum_hess[0,0] = -50 / 1² = -50
     * wsum_hess[1,1] = 0
     * wsum_hess[2,2] = -28 / 3² = -28/9
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    expr *x = new_variable(3, 1, 0, 3);

    /* Create sparse matrix A in CSR format */
    CSR_Matrix *A = new_csr_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *log_x = new_log(x);
    expr *A_log_x = new_left_matmul(log_x, A);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->wsum_hess_init(A_log_x);
    A_log_x->eval_wsum_hess(A_log_x, w);

    /* Expected wsum_hess: diagonal matrix with all 3 entries
     * (sparsity matches child's diagonal Hessian) */
    double expected_x[3] = {
        -50.0,      /* position [0,0]: -50 / 1² */
        -0.0,       /* position [1,1]: 0 * (-1/4) = 0 */
        -28.0 / 9.0 /* position [2,2]: -28 / 9 */
    };
    int expected_i[3] = {0, 1, 2};
    int expected_p[4] = {0, 1, 2, 3}; /* each row has 1 diagonal entry */

    mu_assert("vals incorrect",
              cmp_double_array(A_log_x->wsum_hess->x, expected_x, 3));
    mu_assert("cols incorrect", cmp_int_array(A_log_x->wsum_hess->i, expected_i, 3));
    mu_assert("rows incorrect", cmp_int_array(A_log_x->wsum_hess->p, expected_p, 4));

    free_csr_matrix(A);
    free_expr(A_log_x);
    return 0;
}

const char *test_wsum_hess_left_matmul_composite()
{
    /* Test weighted sum of Hessian of A @ log(B @ x) where:
     * x is 3x1 variable at x = [1, 2, 3]
     * B is 3x3 dense matrix of all ones
     * A is 4x3 sparse matrix [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     * Weights w = [1, 2, 3, 4]
     *
     * B @ x = [6, 6, 6]^T
     * log(B @ x) = [log(6), log(6), log(6)]^T
     * f = A @ log(B @ x) is 4x1
     *
     * The Hessian is more complex due to the composite chain rule.
     * df/dx = A @ diag(1/(B@x)) @ B
     *
     * The second derivative involves:
     * d²f/dx² = A @ d(diag(1/y))/dy @ dy/dx @ B + A @ diag(1/y) @ 0
     * where y = B @ x
     *
     * d(diag(1/y))/dy = -diag(1/y²)
     * dy/dx = B
     *
     * So: d²f[k]/dx² = -sum_j sum_i A[k,i] * (1/(B@x)_i²) * B[i,j] * B[i,l]
     * This is dense and symmetric.
     *
     * For our case with B = all ones, B@x = [6,6,6], so 1/(B@x)² = 1/36 for each
     * element
     *
     * For wsum_hess = sum_k w_k * d²f_k/dx²
     * We use the chain rule: weight w goes through A
     * weighted_w_prime[i] = sum_k w_k * A[k,i]
     * Then: wsum_hess[i,l] = -weighted_w_prime[i] * (1/36) * B[i,j] * (same for l)
     *
     * A^T @ w = [50, 0, 28]^T (from A^T w)
     * So weighted contribution for each element of B@x is -[50, 0, 28] / 36
     *
     * The result is a 3x3 matrix (outer product of weighted_w_prime with B):
     * wsum_hess = -diag([50/36, 0, 28/36]) @ B @ B^T = -diag([50/36, 0, 28/36]) @
     * (all ones) = [-50/36, -50/36, -50/36; 0,      0,      0; -28/36, -28/36,
     * -28/36] = [-25/18, -25/18, -25/18; 0,      0,      0; -7/9,   -7/9,   -7/9]
     *
     * Stored as dense 3x3:
     * nnz = 6 (zeros in row 1)
     * p = [0, 3, 3, 6]
     * i = [0, 1, 2, 0, 1, 2]
     * x = [-25/18, -25/18, -25/18, -7/9, -7/9, -7/9]
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    expr *x = new_variable(3, 1, 0, 3);

    /* Create B matrix (3x3 all ones) */
    CSR_Matrix *B = new_csr_matrix(3, 3, 9);
    int B_p[4] = {0, 3, 6, 9};
    int B_i[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double B_x[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    memcpy(B->p, B_p, 4 * sizeof(int));
    memcpy(B->i, B_i, 9 * sizeof(int));
    memcpy(B->x, B_x, 9 * sizeof(double));

    /* Create A matrix */
    CSR_Matrix *A = new_csr_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *Bx = new_linear(x, B, NULL);
    expr *log_Bx = new_log(Bx);
    expr *A_log_Bx = new_left_matmul(log_Bx, A);

    A_log_Bx->forward(A_log_Bx, x_vals);
    A_log_Bx->jacobian_init(A_log_Bx);
    A_log_Bx->wsum_hess_init(A_log_Bx);
    A_log_Bx->eval_wsum_hess(A_log_Bx, w);

    /* Expected wsum_hess: 3x3 dense matrix (9 nonzeros)
     * All entries are -13/6 because the hessian of log(B@x) w.r.t. x
     * becomes dense when B is dense (all ones) */
    double expected_x[9] = {
        -13.0 / 6.0, -13.0 / 6.0, -13.0 / 6.0, /* row 0 */
        -13.0 / 6.0, -13.0 / 6.0, -13.0 / 6.0, /* row 1 */
        -13.0 / 6.0, -13.0 / 6.0, -13.0 / 6.0  /* row 2 */
    };
    int expected_i[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int expected_p[4] = {0, 3, 6, 9}; /* each row has 3 entries */

    mu_assert("vals incorrect",
              cmp_double_array(A_log_Bx->wsum_hess->x, expected_x, 9));
    mu_assert("cols incorrect",
              cmp_int_array(A_log_Bx->wsum_hess->i, expected_i, 9));
    mu_assert("rows incorrect",
              cmp_int_array(A_log_Bx->wsum_hess->p, expected_p, 4));

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_expr(A_log_Bx);
    return 0;
}

const char *test_wsum_hess_left_matmul_matrix()
{
    /* Test weighted sum of Hessian of A @ log(x) where:
     * x is 3x2 variable, vectorized column-wise: [1, 2, 3, 4, 5, 6]
     * A is 4x3 sparse matrix [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     * Output: A @ log(x) is 4x2, vectorized to 8x1
     * Weights w = [1, 2, 3, 4, 5, 6, 7, 8]
     *
     * The operation is block-diagonal:
     * - Column 0 of x: [1, 2, 3] with weights w[0:4] = [1, 2, 3, 4]
     * - Column 1 of x: [4, 5, 6] with weights w[4:8] = [5, 6, 7, 8]
     *
     * For column 0 (variables x[0:3]):
     * A^T @ w[0:4] = [1*1 + 3*2 + 5*3 + 7*4, 0, 2*1 + 4*2 + 6*3]
     *              = [50, 0, 28]
     * wsum_hess[0,0] = -50 / 1² = -50
     * wsum_hess[1,1] = 0
     * wsum_hess[2,2] = -28 / 3² = -28/9
     *
     * For column 1 (variables x[3:6]):
     * A^T @ w[4:8] = [1*5 + 3*6 + 5*7 + 7*8, 0, 2*5 + 4*6 + 6*7]
     *              = [114, 0, 76]
     * wsum_hess[3,3] = -114 / 4² = -114/16 = -57/8
     * wsum_hess[4,4] = 0
     * wsum_hess[5,5] = -76 / 6² = -76/36 = -19/9
     */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    expr *x = new_variable(3, 2, 0, 6);

    /* Create sparse matrix A in CSR format */
    CSR_Matrix *A = new_csr_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *log_x = new_log(x);
    expr *A_log_x = new_left_matmul(log_x, A);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->wsum_hess_init(A_log_x);
    A_log_x->eval_wsum_hess(A_log_x, w);

    /* Expected wsum_hess: 6x6 diagonal matrix with all 6 entries */
    double expected_x[6] = {
        -50.0,       /* position [0,0]: column 0, variable 0 */
        -0.0,        /* position [1,1]: column 0, variable 1 */
        -28.0 / 9.0, /* position [2,2]: column 0, variable 2 */
        -57.0 / 8.0, /* position [3,3]: column 1, variable 0 */
        -0.0,        /* position [4,4]: column 1, variable 1 */
        -19.0 / 9.0  /* position [5,5]: column 1, variable 2 */
    };
    int expected_i[6] = {0, 1, 2, 3, 4, 5};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6}; /* each row has 1 diagonal entry */

    mu_assert("vals incorrect",
              cmp_double_array(A_log_x->wsum_hess->x, expected_x, 6));
    mu_assert("cols incorrect", cmp_int_array(A_log_x->wsum_hess->i, expected_i, 6));
    mu_assert("rows incorrect", cmp_int_array(A_log_x->wsum_hess->p, expected_p, 7));

    free_csr_matrix(A);
    free_expr(A_log_x);
    return 0;
}

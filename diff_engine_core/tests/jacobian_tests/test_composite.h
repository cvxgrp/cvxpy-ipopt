#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_composite_log()
{
    double u_vals[6] = {0, 0, 1, 2, 3, 0};

    CSR_Matrix *A = new_csr_matrix(2, 6, 6);
    double Ax[6] = {3, 2, 1, 2, 1, 1};
    int Ai[6] = {2, 3, 4, 2, 3, 4};
    int Ap[3] = {0, 3, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    expr *u = new_variable(3, 1, 2, 6);
    expr *Au = new_linear(u, A, NULL);
    expr *log_node = new_log(Au);
    log_node->forward(log_node, u_vals);
    log_node->jacobian_init(log_node);
    log_node->eval_jacobian(log_node);
    double vals[6] = {3.0 / 10, 2.0 / 10, 1.0 / 10, 2.0 / 7, 1.0 / 7, 1.0 / 7};
    int rows[3] = {0, 3, 6};
    int cols[6] = {2, 3, 4, 2, 3, 4};
    mu_assert("vals fail", cmp_double_array(log_node->jacobian->x, vals, 6));
    mu_assert("rows fail", cmp_int_array(log_node->jacobian->p, rows, 3));
    mu_assert("cols fail", cmp_int_array(log_node->jacobian->i, cols, 6));
    free_expr(log_node);
    free_csr_matrix(A);
    return 0;
}

/* u = (z, x, y) where z is 2 x 1, x is 3 x 1, y is 2 x 1,
  f(u) = log(A x) + log(B y)
  where globally

  A = [0 0 1 1 1 0 0
       0 0 2 2 2 0 0
       0 0 3 3 3 0 0]
  B = [0 0 0 0 0 1 1
       0 0 0 0 0 2 2
       0 0 0 0 0 3 3]
*/
const char *test_jacobian_composite_log_add()
{
    double u_vals[7] = {0, 0, 1, 1, 1, 2, 2};

    // create A
    CSR_Matrix *A = new_csr_matrix(3, 7, 9);
    double Ax[9] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    int Ai[9] = {2, 3, 4, 2, 3, 4, 2, 3, 4};
    int Ap[4] = {0, 3, 6, 9};
    memcpy(A->x, Ax, 9 * sizeof(double));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    // create B
    CSR_Matrix *B = new_csr_matrix(3, 7, 6);
    double Bx[6] = {1, 1, 2, 2, 3, 3};
    int Bi[6] = {5, 6, 5, 6, 5, 6};
    int Bp[4] = {0, 2, 4, 6};
    memcpy(B->x, Bx, 6 * sizeof(double));
    memcpy(B->i, Bi, 6 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    expr *x = new_variable(3, 1, 2, 7);
    expr *y = new_variable(2, 1, 5, 7);
    expr *Ax_expr = new_linear(x, A, NULL);
    expr *By_expr = new_linear(y, B, NULL);
    expr *log_Ax = new_log(Ax_expr);
    expr *log_By = new_log(By_expr);
    expr *sum = new_add(log_Ax, log_By);

    sum->forward(sum, u_vals);
    sum->jacobian_init(sum);
    sum->eval_jacobian(sum);

    double vals[15] = {1 / 3.0, 1 / 3.0, 1 / 3.0, 1 / 4.0, 1 / 4.0,
                       1 / 3.0, 1 / 3.0, 1 / 3.0, 1 / 4.0, 1 / 4.0,
                       1 / 3.0, 1 / 3.0, 1 / 3.0, 1 / 4.0, 1 / 4.0};
    int rows[4] = {0, 5, 10, 15};
    int cols[15] = {2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6};

    mu_assert("vals fail", cmp_double_array(sum->jacobian->x, vals, 15));
    mu_assert("rows fail", cmp_int_array(sum->jacobian->p, rows, 4));
    mu_assert("cols fail", cmp_int_array(sum->jacobian->i, cols, 15));
    free_expr(sum);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

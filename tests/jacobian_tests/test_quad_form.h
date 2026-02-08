#include "affine.h"

#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

const char *test_quad_form()
{
    // x^T Q x where x is 3 x 1 variable and has global index 2,
    // Q = [1 2 0; 2 3 0; 0 0 4]
    double u_vals[5] = {0, 0, 1, 2, 3};
    expr *x = new_variable(3, 1, 2, 5);
    CSR_Matrix *Q = new_csr_matrix(3, 3, 5);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));
    expr *node = new_quad_form(x, Q);

    node->jacobian_init(node);
    node->forward(node, u_vals);
    node->eval_jacobian(node);

    double expected_Ax[3] = {10.0, 16.0, 24.0};
    int expected_Ap[2] = {0, 3};
    int expected_Ai[3] = {2, 3, 4};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 3));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 3));
    free_expr(node);
    free_csr_matrix(Q);
    return 0;
}

/* This test is commented out, see the function eval_jabobian_old in
src/other/quad_form.c. const char *test_quad_form2()
{
    // (Au)^T Q (Au) where u is 6 x 1,
    //    Q = [1 2 0;
    //         2 3 0;
    //         0 0 4]
    //    A = [1 0 1 2 3 0;
    //         0 0 4 5 6 0;
    //         1 0 0 2 0 1]
double u_vals[6] = {1, 2, 3, 4, 5, 6};
expr *u = new_variable(6, 1, 0, 6);
CSR_Matrix *Q = new_csr_matrix(3, 3, 5);
double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
int Qi[5] = {0, 1, 0, 1, 2};
int Qp[4] = {0, 2, 4, 5};
memcpy(Q->x, Qx, 5 * sizeof(double));
memcpy(Q->i, Qi, 5 * sizeof(int));
memcpy(Q->p, Qp, 4 * sizeof(int));

CSR_Matrix *A = new_csr_matrix(3, 6, 10);
double Ax[10] = {1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6, 1.0, 2.0, 1.0};
int Ai[10] = {0, 2, 3, 4, 2, 3, 4, 0, 3, 5};
int Ap[4] = {0, 4, 7, 10};
memcpy(A->x, Ax, 10 * sizeof(double));
memcpy(A->i, Ai, 10 * sizeof(int));
memcpy(A->p, Ap, 4 * sizeof(int));
expr *Au = new_linear(u, A, NULL);
expr *node = new_quad_form(Au, Q);

node->jacobian_init(node);
node->forward(node, u_vals);
node->eval_jacobian(node);

double expected_Ax[5] = {422, 2222, 3244, 3786, 120};
int expected_Ap[2] = {0, 5};
int expected_Ai[5] = {0, 2, 3, 4, 5};

mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 5));
mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 5));
free_expr(node);
free_expr(Au);
free_csr_matrix(Q);
free_csr_matrix(A);
return 0;
}
*/

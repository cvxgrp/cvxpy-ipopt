#include "affine.h"

#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

const char *test_quad_over_lin1()
{
    // var = (z, x, w, y) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 1 x 1
    // we compute jacobian of x^T x / y
    double u_vals[10] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0};
    expr *x = new_variable(3, 1, 2, 8);
    expr *y = new_variable(1, 1, 7, 8);
    expr *node = new_quad_over_lin(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double expected_Ax[4] = {2.0 / 4.0, 4.0 / 4.0, 6.0 / 4.0, -14.0 / 16.0};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 3, 4, 7};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 4));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 4));
    free_expr(node);
    return 0;
}

const char *test_quad_over_lin2()
{
    // var = (z, y, w, x) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 1 x 1
    // we compute jacobian of x^T x / y
    double u_vals[10] = {0, 0, 4, 0, 0, 1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 5, 8);
    expr *y = new_variable(1, 1, 2, 8);
    expr *node = new_quad_over_lin(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double expected_Ax[4] = {-14.0 / 16.0, 2.0 / 4.0, 4.0 / 4.0, 6.0 / 4.0};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 5, 6, 7};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 4));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 4));
    free_expr(node);
    return 0;
}

const char *test_quad_over_lin3()
{
    // var = (z, x, w, y) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 1 x 1
    // we compute jacobian of (Ax)^T(Ax)/y where
    // A = [0 0 1 2 3 0 0 0
    //      0 0 4 5 6 0 0]

    CSR_Matrix *A = new_csr_matrix(2, 8, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {2, 3, 4, 2, 3, 4};
    int Ap[3] = {0, 3, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    // Create variables with global indices
    expr *x = new_variable(3, 1, 2, 8);
    expr *y = new_variable(1, 1, 7, 8);
    expr *Ax_expr = new_linear(x, A, NULL);
    expr *node = new_quad_over_lin(Ax_expr, y);
    double u_vals[8] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0};

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double expected_vals[4] = {71.0, 94.0, 117.0, -76.25};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 3, 4, 7};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_vals, 4));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 4));

    free_csr_matrix(A);
    free_expr(node);
    return 0;
}

const char *test_quad_over_lin4()
{
    // var = (z, y, w, x) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 1 x 1
    // we compute jacobian of (Ax)^T(Ax)/y where
    // A = [0 0 0 0 0 1 2 3
    //      0 0 0 0 0 4 5 6
    //

    CSR_Matrix *A = new_csr_matrix(2, 8, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {5, 6, 7, 5, 6, 7};
    int Ap[3] = {0, 3, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    // Create variables with global indices
    expr *x = new_variable(3, 1, 5, 8);
    expr *y = new_variable(1, 1, 2, 8);
    expr *Ax_expr = new_linear(x, A, NULL);
    expr *node = new_quad_over_lin(Ax_expr, y);
    double u_vals[8] = {0, 0, 4, 0, 0, 1.0, 2.0, 3.0};

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double expected_vals[4] = {-76.25, 71.0, 94.0, 117.0};
    int expected_Ap[2] = {0, 4};
    int expected_Ai[4] = {2, 5, 6, 7};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_vals, 4));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 4));

    free_csr_matrix(A);
    free_expr(node);
    return 0;
}

const char *test_quad_over_lin5()
{
    // var = (z, y, w, x) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 1 x 1
    // we compute jacobian of (Avar)^T(Avar)/y where
    // A = [1 0 0 3 0 1 2 3
    //      0 2 0 0 0 4 5 6
    //

    CSR_Matrix *A = new_csr_matrix(2, 8, 9);
    double Ax[9] = {1, 3, 1.0, 2.0, 3.0, 2, 4.0, 5.0, 6.0};
    int Ai[9] = {0, 3, 5, 6, 7, 1, 5, 6, 7};
    int Ap[3] = {0, 5, 9};
    memcpy(A->x, Ax, 9 * sizeof(double));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    // Create variables with global indices
    expr *u = new_variable(8, 1, 0, 8);
    expr *y = new_variable(1, 1, 2, 8);
    expr *Ax_expr = new_linear(u, A, NULL);
    expr *node = new_quad_over_lin(Ax_expr, y);
    double u_vals[8] = {1, 2, 4, 3, 2, 1.0, 2.0, 3.0};

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double expected_vals[7] = {12, 36, -117, 36, 84, 114, 144};
    int expected_Ap[2] = {0, 7};
    int expected_Ai[7] = {0, 1, 2, 3, 5, 6, 7};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_vals, 7));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 7));

    free_csr_matrix(A);
    free_expr(node);
    return 0;
}

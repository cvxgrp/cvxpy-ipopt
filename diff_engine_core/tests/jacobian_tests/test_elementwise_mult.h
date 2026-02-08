#include "affine.h"
#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_jacobian_elementwise_mult_1()
{
    // var = (z, x, w, y) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 3 x 1
    // we compute jacobian of x * y (elementwise multiplication)
    double u_vals[10] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 1, 2, 10);
    expr *y = new_variable(3, 1, 7, 10);
    expr *node = new_elementwise_mult(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double vals[6] = {y->value[0], x->value[0], y->value[1],
                      x->value[1], y->value[2], x->value[2]};
    int rows[4] = {0, 2, 4, 6};
    int cols[6] = {2, 7, 3, 8, 4, 9};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, vals, 6));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, rows, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, cols, 6));
    free_expr(node);
    return 0;
}

const char *test_jacobian_elementwise_mult_2()
{
    // var = (z, y, w, x) where z is 2 x 1, y is 3 x 1, w is 2 x 1, x is 3 x 1
    // we compute jacobian of x * y (elementwise multiplication)
    double u_vals[10] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 1, 7, 10);
    expr *y = new_variable(3, 1, 2, 10);
    expr *node = new_elementwise_mult(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double vals[6] = {x->value[0], y->value[0], x->value[1],
                      y->value[1], x->value[2], y->value[2]};
    int rows[4] = {0, 2, 4, 6};
    int cols[6] = {2, 7, 3, 8, 4, 9};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, vals, 6));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, rows, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, cols, 6));
    free_expr(node);
    return 0;
}

const char *test_jacobian_elementwise_mult_3()
{
    // var = (z, x, w, y) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 3 x 1
    // we compute jacobian of Ax * By (elementwise multiplication)

    /*
    A = [0 0 1 2 0 0 0 0 0 0
         0 0 1 1 3 0 0 0 0 0
         0 0 1 -1 1 0 0 0 0 0]
    */
    CSR_Matrix *A = new_csr_matrix(3, 10, 9);
    double Ax_vals[9] = {1.0, 2.0, 1.0, 1.0, 3.0, 1.0, -1.0, 1.0};
    int Ai[9] = {2, 3, 2, 3, 4, 2, 3, 4};
    int Ap[4] = {0, 2, 5, 8};
    memcpy(A->x, Ax_vals, 9 * sizeof(double));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /*
    B = [0 0 0 0 0 0 0 1 3 0
         0 0 0 0 0 0 0 1 1 4
         0 0 0 0 0 0 0 1 -2 1]

    */
    CSR_Matrix *B = new_csr_matrix(3, 10, 9);
    double Bx_vals[9] = {1.0, 3.0, 1.0, 1.0, 4.0, 1.0, -2.0, 1.0};
    int Bi[9] = {7, 8, 7, 8, 9, 7, 8, 9};
    int Bp[4] = {0, 2, 5, 8};
    memcpy(B->x, Bx_vals, 9 * sizeof(double));
    memcpy(B->i, Bi, 9 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    double u_vals[10] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 1, 2, 10);
    expr *y = new_variable(3, 1, 7, 10);
    expr *Ax = new_linear(x, A, NULL);
    expr *By = new_linear(y, B, NULL);
    expr *node = new_elementwise_mult(Ax, By);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    /* Correct answer:
      0 0 19 38 0  0 0 5  15  0
      0 0 33 33 99 0 0 12 12 48
      0 0  a  a  a 0 0 2  -4 2

    where a is an explicit 0
    */

    double vals[16] = {19.0, 38.0, 5.0, 15.0, 33.0, 33.0, 99.0, 12.0,
                       12.0, 48.0, 0.0, 0.0,  0.0,  2.0,  -4.0, 2.0};
    int rows[4] = {0, 4, 10, 16};
    int cols[16] = {2, 3, 7, 8, 2, 3, 4, 7, 8, 9, 2, 3, 4, 7, 8, 9};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, vals, 16));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, rows, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, cols, 16));
    free_expr(node);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_jacobian_elementwise_mult_4()
{
    // var = (z, x, w, y) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 3 x 1
    // we compute jacobian of Ax * Ax (elementwise multiplication)

    /*
    A = [0 0 1 2 0 0 0 0 0 0
         0 0 1 1 3 0 0 0 0 0
         0 0 1 -1 1 0 0 0 0 0]
    */
    CSR_Matrix *A = new_csr_matrix(3, 10, 9);
    double Ax_vals[9] = {1.0, 2.0, 1.0, 1.0, 3.0, 1.0, -1.0, 1.0};
    int Ai[9] = {2, 3, 2, 3, 4, 2, 3, 4};
    int Ap[4] = {0, 2, 5, 8};
    memcpy(A->x, Ax_vals, 9 * sizeof(double));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    double u_vals[10] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 1, 2, 10);
    expr *Ax = new_linear(x, A, NULL);
    expr *node = new_elementwise_mult(Ax, Ax);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    /* Correct answer: 0 0 10 20 0  0 0 0 0 0
                       0 0 24 24 72 0 0 0 0 0
                       0 0 4  -4 4  0 0 0 0 0
    */

    double vals[8] = {10, 20, 24, 24, 72, 4, -4, 4};
    int rows[4] = {0, 2, 5, 8};
    int cols[8] = {2, 3, 2, 3, 4, 2, 3, 4};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, vals, 8));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, rows, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, cols, 8));
    free_expr(node);
    free_csr_matrix(A);
    return 0;
}

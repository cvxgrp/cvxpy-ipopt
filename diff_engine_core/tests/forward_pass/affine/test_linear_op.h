#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_linear_op()
{
    /* create CSR matrix
     A = [0 0 2 3 0 0]
         [0 0 1 0 2 0]
         [0 0 3 4 5 0] */
    double Ax[7] = {2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    int Ai[7] = {2, 3, 2, 4, 2, 3, 4};
    int Ap[4] = {0, 2, 4, 7};
    CSR_Matrix *A = new_csr_matrix(3, 6, 7);
    memcpy(A->x, Ax, 7 * sizeof(double));
    memcpy(A->i, Ai, 7 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    expr *var = new_variable(3, 1, 2, 6);
    expr *linear_node = new_linear(var, A, NULL);
    double x[6] = {0, 0, 1, 2, 3, 0};
    linear_node->forward(linear_node, x);

    double expected[3] = {8, 7, 26};
    mu_assert("fail", cmp_double_array(linear_node->value, expected, 3));
    free_expr(linear_node);
    free_csr_matrix(A);
    return 0;
}

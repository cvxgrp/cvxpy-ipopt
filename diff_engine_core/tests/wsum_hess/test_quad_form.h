#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"
#include <string.h>

const char *test_wsum_hess_quad_form()
{
    // x has var_id = 3, dimension 4, total variables = 10
    double u_vals[10] = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0};
    double w = 2.0;

    /* Symmetric 4x4 Q:
     * [1 2 0 0]
     * [2 5 3 0]
     * [0 3 4 1]
     * [0 0 1 6]
     */
    CSR_Matrix *Q = new_csr_matrix(4, 4, 10);
    double Qx[10] = {1.0, 2.0, 2.0, 5.0, 3.0, 3.0, 4.0, 1.0, 1.0, 6.0};
    int Qi[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    int Qp[5] = {0, 2, 5, 8, 10};
    memcpy(Q->x, Qx, 10 * sizeof(double));
    memcpy(Q->i, Qi, 10 * sizeof(int));
    memcpy(Q->p, Qp, 5 * sizeof(int));

    expr *x = new_variable(4, 1, 3, 10);
    expr *node = new_quad_form(x, Q);

    node->jacobian_init(node);
    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, &w);

    int expected_p[11] = {0, 0, 0, 0, 2, 5, 8, 10, 10, 10, 10};
    int expected_i[10] = {3, 4, 3, 4, 5, 4, 5, 6, 5, 6};
    double expected_x[10] = {4.0, 8.0, 8.0, 20.0, 12.0, 12.0, 16.0, 4.0, 4.0, 24.0};

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 11));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 10));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 10));

    free_expr(node);
    free_csr_matrix(Q);
    return 0;
}

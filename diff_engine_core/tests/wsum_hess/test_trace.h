#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_trace_variable()
{
    /* Test weighted sum of Hessian of trace(x) where x is 3x3 variable
     * x has global variable index 1
     * Total 13 variables
     * x.value = [1, 2, 3, 4, 5, 6, 7, 8, 9] (column-wise)
     * Weight = 2.0
     *
     * trace(x) = x[0,0] + x[1,1] + x[2,2] = 1 + 5 + 9 = 15
     * Since x is linear (identity mapping), Hessian is zero.
     * wsum_hess should be a 13x13 zero sparse matrix (or empty)
     */
    double u_vals[13] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 0.0, 0.0, 0.0};
    double w = 2.0;

    expr *x = new_variable(3, 3, 1, 13);
    expr *trace_node = new_trace(x);

    trace_node->forward(trace_node, u_vals);
    trace_node->jacobian_init(trace_node);
    trace_node->wsum_hess_init(trace_node);
    trace_node->eval_wsum_hess(trace_node, &w);

    /* For a linear operation (variable), Hessian is zero */
    mu_assert("wsum_hess should be empty", trace_node->wsum_hess->nnz == 0);

    mu_assert("dims correct",
              trace_node->wsum_hess->m == 13 && trace_node->wsum_hess->n == 13);

    free_expr(trace_node);
    return 0;
}

const char *test_wsum_hess_trace_log_variable()
{
    /* Test weighted sum of Hessian of trace(log(x)) where x is 3x3 variable
     * x has global variable index 1
     * Total 13 variables
     * trace(log(x)) = log(x[0,0]) + log(x[1,1]) + log(x[2,2])
     * Hessian is diagonal,  BUT NOTE THAT THIS STORES STRUCTURAL ZEROS FOR ALL
     * NON-DIAGONAL ENTRIES
     */
    double u_vals[13] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 0.0, 0.0, 0.0};
    double w = 2.0;

    expr *x = new_variable(3, 3, 1, 13);
    expr *log_node = new_log(x);
    expr *trace_node = new_trace(log_node);

    trace_node->forward(trace_node, u_vals);
    trace_node->jacobian_init(trace_node);
    trace_node->wsum_hess_init(trace_node);
    trace_node->eval_wsum_hess(trace_node, &w);

    double expected_Ax[9] = {-2.0, 0, 0, 0, -0.08, 0, 0, 0, -0.024691358024691357};
    int expected_Ap[14] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9};
    int expected_Ai[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    CSR_Matrix *H = trace_node->wsum_hess;
    mu_assert("nnz wrong", H->nnz == 9);
    mu_assert("vals match", cmp_double_array(H->x, expected_Ax, 9));
    mu_assert("cols match", cmp_int_array(H->i, expected_Ai, 9));
    mu_assert("rows fail", cmp_int_array(H->p, expected_Ap, 14));

    free_expr(trace_node);
    return 0;
}

const char *test_wsum_hess_trace_composite()
{
    /* Test weighted sum of Hessian of trace(log(x) + exp(x)) where x is 3x3 variable
     * x has global variable index 1
     * Total 13 variables
     * x.value = [1, 2, 3, 4, 5, 6, 7, 8, 9] (column-wise)
     * Weight = 2.0
     *
     * f(x) = trace(log(x) + exp(x))
     * d/dx (log(x) + exp(x)) = 1/x + exp(x)
     * d²/dx² (log(x) + exp(x)) = -1/x² + exp(x)
     *
     * For diagonal elements:
     * x[0,0] = 1: d²/dx² = -1/1² + exp(1) = -1 + 2.718... ≈ 1.718...
     * x[1,1] = 5: d²/dx² = -1/25 + exp(5) = -0.04 + 148.413... ≈ 148.373...
     * x[2,2] = 9: d²/dx² = -1/81 + exp(9) ≈ -0.0123... + 8103.08... ≈ 8103.07...
     *
     * wsum_hess (13x13) has nonzeros only at diagonal positions [1,1], [5,5], [9,9]
     * with values = weight * second derivatives = 2 * values above
     */
    double u_vals[13] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 0.0, 0.0, 0.0};
    double w = 2.0;

    expr *x = new_variable(3, 3, 1, 13);
    expr *log_node = new_log(x);
    expr *exp_node = new_exp(x);
    expr *add_node = new_add(log_node, exp_node);
    expr *trace_node = new_trace(add_node);

    trace_node->forward(trace_node, u_vals);
    trace_node->jacobian_init(trace_node);
    trace_node->wsum_hess_init(trace_node);
    trace_node->eval_wsum_hess(trace_node, &w);

    /* Expected diagonal Hessian values at indices [1,1], [5,5], [9,9]
     * d²(log(x_ii) + exp(x_ii))/dx_ii² = -1/x_ii² + exp(x_ii)
     * At x_00 = 1: -1 + exp(1) = -1 + 2.718281828...
     * At x_11 = 5: -1/25 + exp(5) = -0.04 + 148.413159103...
     * At x_22 = 9: -1/81 + exp(9) ≈ -0.01234... + 8103.083927576...
     *
     * wsum_hess values = 2 * second derivatives
     */
    double exp_1 = 2.718281828;
    double exp_5 = 148.413159103;
    double exp_9 = 8103.083927576;

    double d2_1 = -1.0 + exp_1;
    double d2_5 = -0.04 + exp_5;
    double d2_9 = -1.0 / 81.0 + exp_9;

    double expected_Ax[9] = {w * d2_1, 0, 0, 0, w * d2_5, 0, 0, 0, w * d2_9};
    int expected_Ap[14] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9};
    int expected_Ai[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    mu_assert("nnz wrong", trace_node->wsum_hess->nnz == 9);
    mu_assert("rows fail", cmp_int_array(trace_node->wsum_hess->p, expected_Ap, 14));
    mu_assert("vals match",
              cmp_double_array(trace_node->wsum_hess->x, expected_Ax, 9));
    mu_assert("cols match", cmp_int_array(trace_node->wsum_hess->i, expected_Ai, 9));
    free_expr(trace_node);
    return 0;
}

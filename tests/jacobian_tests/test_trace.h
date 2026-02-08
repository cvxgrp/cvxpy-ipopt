#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_trace_variable()
{
    /* Test Jacobian of trace(x) where x is 3x3 variable
     * x has global variable index 1
     * Total 13 variables
     * x.value = [1, 2, 3, 4, 5, 6, 7, 8, 9] (column-wise)
     * Stored as:
     * [[1, 4, 7],
     *  [2, 5, 8],
     *  [3, 6, 9]]
     * Diagonal: [1, 5, 9]
     * trace(x) = 1 + 5 + 9 = 15
     *
     * Jacobian: d(trace)/dx has nonzeros only at diagonal positions
     * Indices (0-indexed, column-wise): 0, 4, 8
     * So global variable indices: 1, 5, 9 (offset by 1)
     * Values: [1, 1, 1]
     * Result is 1x13 sparse matrix with 3 nonzeros
     */
    double u_vals[13] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 0.0, 0.0, 0.0};
    expr *x = new_variable(3, 3, 1, 13);
    expr *trace_node = new_trace(x);

    trace_node->forward(trace_node, u_vals);
    trace_node->jacobian_init(trace_node);
    trace_node->eval_jacobian(trace_node);

    double expected_Ax[3] = {1.0, 1.0, 1.0};
    int expected_Ap[2] = {0, 3};
    int expected_Ai[3] = {1, 5, 9}; /* column indices (global variable indices) */

    mu_assert("vals fail",
              cmp_double_array(trace_node->jacobian->x, expected_Ax, 3));
    mu_assert("rows fail", cmp_int_array(trace_node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(trace_node->jacobian->i, expected_Ai, 3));

    free_expr(trace_node);
    return 0;
}

const char *test_jacobian_trace_composite()
{
    /* Test Jacobian of trace(log(x) + exp(x)) where x is 3x3 variable
     * x has global variable index 1
     * Total 13 variables
     * x.value = [1, 2, 3, 4, 5, 6, 7, 8, 9] (column-wise)
     * Diagonal elements: x[0, 0] = 1, x[1, 1] = 5, x[2, 2] = 9
     *
     * log(x) + exp(x) elementwise: [log(x_ij) + exp(x_ij)]
     * At diagonal: [log(1) + exp(1), log(5) + exp(5), log(9) + exp(9)]
     *
     * d/dx (log(x) + exp(x)) = [1/x + exp(x)]
     * At diagonal elements:
     * x[0,0] = 1: 1/1 + exp(1) ≈ 1 + 2.718... ≈ 3.718...
     * x[1,1] = 5: 1/5 + exp(5) = 0.2 + 148.413... ≈ 148.613...
     * x[2,2] = 9: 1/9 + exp(9) ≈ 0.111... + 8103.08... ≈ 8103.19...
     *
     * Jacobian (1x13 sparse):
     * Nonzeros at columns [1, 5, 9] with values above
     * We'll compute and check the actual values
     */
    double u_vals[13] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 0.0, 0.0, 0.0};
    expr *x = new_variable(3, 3, 1, 13);
    expr *log_node = new_log(x);
    expr *exp_node = new_exp(x);
    expr *add_node = new_add(log_node, exp_node);
    expr *trace_node = new_trace(add_node);

    trace_node->jacobian_init(trace_node);
    trace_node->forward(trace_node, u_vals);
    trace_node->eval_jacobian(trace_node);

    /* Expected values: d(log(x_ii) + exp(x_ii))/dx_ii = 1/x_ii + exp(x_ii)
     * At x_00 = 1: 1/1 + exp(1) = 1 + 2.718281828...
     * At x_11 = 5: 1/5 + exp(5) = 0.2 + 148.413159103...
     * At x_22 = 9: 1/9 + exp(9) = 0.111... + 8103.083927576...
     */
    double exp_1 = 2.718281828;
    double exp_5 = 148.413159103;
    double exp_9 = 8103.083927576;
    double expected_Ax[3] = {1.0 + exp_1, 0.2 + exp_5, 1.0 / 9.0 + exp_9};
    int expected_Ap[2] = {0, 3};
    int expected_Ai[3] = {1, 5, 9}; /* column indices (global variable indices) */

    mu_assert("vals match count", trace_node->jacobian->nnz == 3);
    mu_assert("rows fail", cmp_int_array(trace_node->jacobian->p, expected_Ap, 2));
    mu_assert("cols fail", cmp_int_array(trace_node->jacobian->i, expected_Ai, 3));
    mu_assert("vals fail",
              cmp_double_array(trace_node->jacobian->x, expected_Ax, 3));

    free_expr(trace_node);
    return 0;
}

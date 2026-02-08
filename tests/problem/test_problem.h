#ifndef TEST_PROBLEM_H
#define TEST_PROBLEM_H

#include <math.h>
#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "test_helpers.h"

/*
 * Test problem: minimize sum(log(x))
 *               subject to x >= 1 (as x - 1 >= 0)
 *
 * With x of size 3, n_vars = 3
 */
const char *test_problem_new_free(void)
{
    /* Create expressions */
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *objective = new_sum(log_x, -1);

    /* Create constraint: x - 1 (represented as just x for simplicity) */
    expr *x_constraint = new_variable(3, 1, 0, 3);

    expr *constraints[1] = {x_constraint};

    /* Create problem */
    problem *prob = new_problem(objective, constraints, 1, true);

    mu_assert("new_problem failed", prob != NULL);
    mu_assert("n_vars wrong", prob->n_vars == 3);
    mu_assert("n_constraints wrong", prob->n_constraints == 1);
    mu_assert("total_constraint_size wrong", prob->total_constraint_size == 3);

    /* Free problem (recursively frees expressions) */
    free_problem(prob);

    return 0;
}

/*
 * Test problem_objective_forward: minimize sum(log(x))
 *                       subject to x (as constraint)
 */
const char *test_problem_objective_forward(void)
{
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *objective = new_sum(log_x, -1);

    expr *x_constraint = new_variable(3, 1, 0, 3);
    expr *constraints[1] = {x_constraint};

    problem *prob = new_problem(objective, constraints, 1, true);

    double u[3] = {1.0, 2.0, 3.0};
    problem_init_derivatives(prob);

    double obj_val = problem_objective_forward(prob, u);

    /* Expected: sum(log([1, 2, 3])) = 0 + log(2) + log(3) */
    double expected_obj = log(1.0) + log(2.0) + log(3.0);
    mu_assert("objective value wrong", fabs(obj_val - expected_obj) < 1e-10);

    /* Now evaluate constraints separately */
    problem_constraint_forward(prob, u);

    /* Constraint values should be [1, 2, 3] */
    double expected_constraints[3] = {1.0, 2.0, 3.0};
    mu_assert("constraint values wrong",
              cmp_double_array(prob->constraint_values, expected_constraints, 3));

    free_problem(prob);

    return 0;
}

/*
 * Test problem_constraint_forward: evaluate constraints only
 *   Constraint 1: log(x) -> [log(2), log(4)]
 *   Constraint 2: exp(x) -> [exp(2), exp(4)]
 */
const char *test_problem_constraint_forward(void)
{
    int n_vars = 2;

    /* Shared variable */
    expr *x = new_variable(2, 1, 0, n_vars);

    /* Objective: sum(log(x)) */
    expr *log_obj = new_log(x);
    expr *objective = new_sum(log_obj, -1);

    /* Constraint 1: log(x) */
    expr *log_c1 = new_log(x);

    /* Constraint 2: exp(x) */
    expr *exp_c2 = new_exp(x);

    expr *constraints[2] = {log_c1, exp_c2};

    problem *prob = new_problem(objective, constraints, 2, true);

    double u[2] = {2.0, 4.0};
    problem_init_derivatives(prob);

    problem_constraint_forward(prob, u);

    /* Check constraint values: [log(2), log(4), exp(2), exp(4)] */
    double expected_constraints[4] = {log(2.0), log(4.0), exp(2.0), exp(4.0)};
    mu_assert("constraint values wrong",
              cmp_double_array(prob->constraint_values, expected_constraints, 4));

    free_problem(prob);

    return 0;
}

/*
 * Test problem_gradient: gradient of sum(log(x)) = [1/x1, 1/x2, 1/x3]
 */
const char *test_problem_gradient(void)
{
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *objective = new_sum(log_x, -1);

    problem *prob = new_problem(objective, NULL, 0, true);

    double u[3] = {1.0, 2.0, 4.0};
    problem_init_derivatives(prob);

    problem_objective_forward(prob, u);
    problem_gradient(prob);

    /* Expected gradient: [1/1, 1/2, 1/4] = [1.0, 0.5, 0.25] */
    double expected_grad[3] = {1.0, 0.5, 0.25};
    mu_assert("gradient wrong",
              cmp_double_array(prob->gradient_values, expected_grad, 3));

    free_problem(prob);

    return 0;
}

/*
 * Test problem_jacobian: one constraint log(x)
 * Jacobian of log(x): diag([1/x1, 1/x2])
 */
const char *test_problem_jacobian(void)
{
    int n_vars = 2;

    /* Create separate expression trees */
    expr *x_obj = new_variable(2, 1, 0, n_vars);
    expr *log_obj = new_log(x_obj);
    expr *objective = new_sum(log_obj, -1);

    expr *x_c1 = new_variable(2, 1, 0, n_vars);
    expr *log_c1 = new_log(x_c1);

    expr *constraints[1] = {log_c1};

    problem *prob = new_problem(objective, constraints, 1, true);

    double u[2] = {2.0, 4.0};
    problem_init_derivatives(prob);

    problem_constraint_forward(prob, u);
    problem_jacobian(prob);

    CSR_Matrix *jac = prob->jacobian;

    /* Check dimensions */
    mu_assert("jac rows wrong", jac->m == 2);
    mu_assert("jac cols wrong", jac->n == 2);

    /* Check row pointers: each row has 1 element */
    int expected_p[3] = {0, 1, 2};
    mu_assert("jac->p wrong", cmp_int_array(jac->p, expected_p, 3));

    /* Check column indices */
    int expected_i[2] = {0, 1};
    mu_assert("jac->i wrong", cmp_int_array(jac->i, expected_i, 2));

    /* Check values: [1/2, 1/4] */
    double expected_x[2] = {0.5, 0.25};
    mu_assert("jac->x wrong", cmp_double_array(jac->x, expected_x, 2));

    free_problem(prob);

    return 0;
}

/*
 * Test problem_jacobian with multiple constraints and SHARED variable:
 *   Constraint 1: log(x) -> Jacobian diag([1/x1, 1/x2])
 *   Constraint 2: exp(x) -> Jacobian diag([exp(x1), exp(x2)])
 *
 * Stacked jacobian (4x2):
 *   [ 1/x1    0    ]
 *   [  0    1/x2   ]
 *   [exp(x1)  0    ]
 *   [  0    exp(x2)]
 *
 * Note: All expressions share the same variable node x, testing that
 * free_problem correctly handles shared nodes without double-free.
 */
const char *test_problem_jacobian_multi(void)
{
    int n_vars = 2;

    /* Single shared variable used across all expressions */
    expr *x = new_variable(2, 1, 0, n_vars);

    /* Objective: sum(log(x)) */
    expr *log_obj = new_log(x);
    expr *objective = new_sum(log_obj, -1);

    /* Constraint 1: log(x) - shares x */
    expr *log_c1 = new_log(x);

    /* Constraint 2: exp(x) - shares x */
    expr *exp_c2 = new_exp(x);

    expr *constraints[2] = {log_c1, exp_c2};

    problem *prob = new_problem(objective, constraints, 2, true);

    double u[2] = {2.0, 4.0};
    problem_init_derivatives(prob);

    problem_constraint_forward(prob, u);
    problem_jacobian(prob);

    CSR_Matrix *jac = prob->jacobian;

    /* Check dimensions: 4 rows (2 + 2), 2 cols */
    mu_assert("jac rows wrong", jac->m == 4);
    mu_assert("jac cols wrong", jac->n == 2);
    mu_assert("jac nnz wrong", jac->nnz == 4);

    /* Check row pointers: each row has 1 element */
    int expected_p[5] = {0, 1, 2, 3, 4};
    mu_assert("jac->p wrong", cmp_int_array(jac->p, expected_p, 5));

    /* Check column indices: diagonal pattern */
    int expected_i[4] = {0, 1, 0, 1};
    mu_assert("jac->i wrong", cmp_int_array(jac->i, expected_i, 4));

    /* Check values:
     * Row 0: 1/2 = 0.5
     * Row 1: 1/4 = 0.25
     * Row 2: exp(2) ≈ 7.389
     * Row 3: exp(4) ≈ 54.598
     */
    double expected_x[4] = {0.5, 0.25, exp(2.0), exp(4.0)};
    mu_assert("jac->x wrong", cmp_double_array(jac->x, expected_x, 4));

    free_problem(prob);

    return 0;
}

/*
 * Test problem_hessian: Lagrange Hessian of:
 *   Objective: sum(log(x)) where x is 3x1
 *   Constraint 1: exp(x)
 *   Constraint 2: sin(x)
 *
 * Evaluate at x = [1, 2, 3] with w = [1, 2, 3, 4, 5, 6], w_obj = 2
 *
 * Lagrange function:
 *   L = 2*sum(log(x)) + w[0:3]^T exp(x) + w[3:6]^T sin(x)
 *
 * Hessian at x = [1, 2, 3]:
 *   H_obj = diag([-1/1^2, -1/2^2, -1/3^2]) = diag([-1, -0.25, -0.111111])
 *   H_c1 = diag([exp(1), exp(2), exp(3)]) (element i has weight w[i])
 *   H_c2 = diag([-sin(1), -sin(2), -sin(3)]) (element i has weight w[i+3])
 *
 * Lagrange Hessian diagonal elements:
 *   H[0,0] = 2*(-1) + 1*exp(1) + 4*(-sin(1))
 *   H[1,1] = 2*(-0.25) + 2*exp(2) + 5*(-sin(2))
 *   H[2,2] = 2*(-0.111111) + 3*exp(3) + 6*(-sin(3))
 */
const char *test_problem_hessian(void)
{
    int n_vars = 3;

    /* Shared variable */
    expr *x = new_variable(3, 1, 0, n_vars);

    /* Objective: sum(log(x)) */
    expr *log_obj = new_log(x);
    expr *objective = new_sum(log_obj, -1);

    /* Constraint 1: exp(x) */
    expr *exp_c1 = new_exp(x);

    /* Constraint 2: sin(x) */
    expr *sin_c2 = new_sin(x);

    expr *constraints[2] = {exp_c1, sin_c2};

    problem *prob = new_problem(objective, constraints, 2, true);

    double u[3] = {1.0, 2.0, 3.0};
    problem_init_derivatives(prob);

    /* Forward pass */
    problem_objective_forward(prob, u);
    problem_constraint_forward(prob, u);

    /* Evaluate Lagrange Hessian */
    double w[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w_obj = 2.0;
    problem_hessian(prob, w_obj, w);

    CSR_Matrix *H = prob->lagrange_hessian;

    /* Check dimensions: 3x3 symmetric */
    mu_assert("H rows wrong", H->m == 3);
    mu_assert("H cols wrong", H->n == 3);

    /* Compute expected diagonal values */
    double expected_H00 = 2.0 * (-1.0) + 1.0 * exp(1.0) + 4.0 * (-sin(1.0));
    double expected_H11 = 2.0 * (-0.25) + 2.0 * exp(2.0) + 5.0 * (-sin(2.0));
    double expected_H22 = 2.0 * (-1.0 / 9.0) + 3.0 * exp(3.0) + 6.0 * (-sin(3.0));

    /* Since Hessian is diagonal, check the diagonal entries
     * Row pointers should be [0, 1, 2, 3] */
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};
    double expected_x[3] = {expected_H00, expected_H11, expected_H22};
    mu_assert("H->p wrong", cmp_int_array(H->p, expected_p, 4));
    mu_assert("H->i wrong", cmp_int_array(H->i, expected_i, 3));
    mu_assert("H->x wrong", cmp_double_array(H->x, expected_x, 3));

    free_problem(prob);

    return 0;
}

#endif /* TEST_PROBLEM_H */

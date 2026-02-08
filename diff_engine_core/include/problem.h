/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PROBLEM_H
#define PROBLEM_H

#include "expr.h"
#include "utils/CSR_Matrix.h"
#include "utils/Timer.h"
#include <stdbool.h>

typedef struct
{
    double time_init_derivatives;
    double time_eval_jacobian;
    double time_eval_gradient;
    double time_eval_hessian;
    double time_forward_obj;
    double time_forward_constraints;

    int nnz_affine;
    int nnz_nonlinear; /* jacobian of nonlinear constraints */
    int nnz_hessian;
} Diff_engine_stats;

typedef struct problem
{
    expr *objective;
    expr **constraints;
    int n_constraints;
    int n_vars;
    int total_constraint_size;

    /* Allocated by new_problem */
    double *constraint_values;
    double *gradient_values;

    /* Allocated by problem_init_derivatives */
    CSR_Matrix *jacobian;
    CSR_Matrix *lagrange_hessian;
    int *hess_idx_map; /* Maps all wsum_hess nnz to lagrange_hessian (obj +
                          constraints) */

    /* for the affine shortcut we keep track of the first time the jacobian and
     * hessian are called */
    bool jacobian_called;

    /* Statistics for performance measurement */
    Diff_engine_stats stats;
    bool verbose;
} problem;

/* Retains objective and constraints (shared ownership with caller) */
problem *new_problem(expr *objective, expr **constraints, int n_constraints,
                     bool verbose);
void problem_init_jacobian(problem *prob);
void problem_init_hessian(problem *prob);
void problem_init_derivatives(problem *prob);
void free_problem(problem *prob);

double problem_objective_forward(problem *prob, const double *u);
void problem_constraint_forward(problem *prob, const double *u);
void problem_gradient(problem *prob);
void problem_jacobian(problem *prob);
void problem_hessian(problem *prob, double obj_w, const double *w);

#endif

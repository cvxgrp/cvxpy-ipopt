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
#include "problem.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* forward declaration */
static void problem_lagrange_hess_fill_sparsity(problem *prob, int *iwork);

problem *new_problem(expr *objective, expr **constraints, int n_constraints,
                     bool verbose)
{
    problem *prob = (problem *) calloc(1, sizeof(problem));
    if (!prob) return NULL;

    /* objective */
    prob->objective = objective;
    expr_retain(objective);
    prob->n_vars = objective->n_vars;
    prob->jacobian_called = false;

    /* constraints array */
    prob->total_constraint_size = 0;
    prob->n_constraints = n_constraints;
    if (n_constraints > 0)
    {
        prob->constraints = (expr **) malloc(n_constraints * sizeof(expr *));
        for (int i = 0; i < n_constraints; i++)
        {
            prob->constraints[i] = constraints[i];
            prob->total_constraint_size += constraints[i]->size;
            expr_retain(constraints[i]);
        }
    }

    /* allocation */
    prob->constraint_values =
        (double *) calloc(prob->total_constraint_size, sizeof(double));
    prob->gradient_values = (double *) calloc(prob->n_vars, sizeof(double));

    /* Initialize statistics */
    prob->stats.time_init_derivatives = 0.0;
    prob->stats.time_eval_jacobian = 0.0;
    prob->stats.time_eval_hessian = 0.0;
    prob->stats.time_forward_obj = 0.0;
    prob->stats.time_forward_constraints = 0.0;
    prob->stats.nnz_affine = 0;
    prob->stats.nnz_nonlinear = 0;

    prob->verbose = verbose;

    return prob;
}

static void problem_lagrange_hess_fill_sparsity(problem *prob, int *iwork)
{
    expr **constrs = prob->constraints;
    int *cols = iwork;
    int *col_to_pos = iwork; /* reused after qsort */
    int nnz = 0;
    CSR_Matrix *H_obj = prob->objective->wsum_hess;
    CSR_Matrix *H_c;
    CSR_Matrix *H = prob->lagrange_hessian;
    H->p[0] = 0;

    // ----------------------------------------------------------------------
    //                      Fill sparsity pattern
    // ----------------------------------------------------------------------
    for (int row = 0; row < H->m; row++)
    {
        /* gather columns from objective hessian */
        int count = H_obj->p[row + 1] - H_obj->p[row];
        memcpy(cols, H_obj->i + H_obj->p[row], count * sizeof(int));

        /* gather columns from constraint hessians */
        for (int c_idx = 0; c_idx < prob->n_constraints; c_idx++)
        {
            H_c = constrs[c_idx]->wsum_hess;
            int c_len = H_c->p[row + 1] - H_c->p[row];
            memcpy(cols + count, H_c->i + H_c->p[row], c_len * sizeof(int));
            count += c_len;
        }

        /* find unique columns */
        sort_int_array(cols, count);
        int prev_col = -1;
        for (int j = 0; j < count; j++)
        {
            if (cols[j] != prev_col)
            {
                H->i[nnz] = cols[j];
                nnz++;
                prev_col = cols[j];
            }
        }

        H->p[row + 1] = nnz;
    }

    H->nnz = nnz;

    // ----------------------------------------------------------------------
    //                           Build idx map
    // ----------------------------------------------------------------------
    int idx_offset = 0;

    /* map objective hessian entries */
    for (int row = 0; row < H->m; row++)
    {
        for (int idx = H->p[row]; idx < H->p[row + 1]; idx++)
        {
            col_to_pos[H->i[idx]] = idx;
        }

        for (int j = H_obj->p[row]; j < H_obj->p[row + 1]; j++)
        {
            prob->hess_idx_map[idx_offset++] = col_to_pos[H_obj->i[j]];
        }
    }

    /* map constraint hessian entries */
    for (int c_idx = 0; c_idx < prob->n_constraints; c_idx++)
    {
        H_c = constrs[c_idx]->wsum_hess;
        for (int row = 0; row < H->m; row++)
        {
            for (int idx = H->p[row]; idx < H->p[row + 1]; idx++)
            {
                col_to_pos[H->i[idx]] = idx;
            }

            for (int j = H_c->p[row]; j < H_c->p[row + 1]; j++)
            {
                prob->hess_idx_map[idx_offset++] = col_to_pos[H_c->i[j]];
            }
        }
    }
}

void problem_init_jacobian(problem *prob)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // -------------------------------------------------------------------------------
    //                           Jacobian structure
    // -------------------------------------------------------------------------------
    prob->objective->jacobian_init(prob->objective);
    int nnz = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->jacobian_init(c);
        nnz += c->jacobian->nnz;

        if (c->is_affine(c))
        {
            prob->stats.nnz_affine += c->jacobian->nnz;
        }
        else
        {
            prob->stats.nnz_nonlinear += c->jacobian->nnz;
        }
    }

    prob->jacobian = new_csr_matrix(prob->total_constraint_size, prob->n_vars, nnz);

    /* set sparsity pattern of jacobian */
    CSR_Matrix *H = prob->jacobian;
    H->p[0] = 0;
    int row_offset = 0;
    int nnz_offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];

        for (int r = 1; r <= c->jacobian->m; r++)
        {
            H->p[row_offset + r] = nnz_offset + c->jacobian->p[r];
        }

        memcpy(H->i + nnz_offset, c->jacobian->i, c->jacobian->nnz * sizeof(int));
        row_offset += c->jacobian->m;
        nnz_offset += c->jacobian->nnz;
    }
    assert(nnz_offset == nnz);

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);
}

void problem_init_hessian(problem *prob)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // -------------------------------------------------------------------------------
    //                        Lagrange Hessian structure
    // -------------------------------------------------------------------------------
    prob->objective->wsum_hess_init(prob->objective);
    int nnz = prob->objective->wsum_hess->nnz;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        prob->constraints[i]->wsum_hess_init(prob->constraints[i]);
        nnz += prob->constraints[i]->wsum_hess->nnz;
    }

    prob->lagrange_hessian = new_csr_matrix(prob->n_vars, prob->n_vars, nnz);
    memset(prob->lagrange_hessian->x, 0, nnz * sizeof(double)); /* affine shortcut */
    prob->stats.nnz_hessian = nnz;
    prob->hess_idx_map = (int *) malloc(nnz * sizeof(int));
    int *iwork = (int *) malloc(MAX(nnz, prob->n_vars) * sizeof(int));
    problem_lagrange_hess_fill_sparsity(prob, iwork);
    free(iwork);

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);
}

void problem_init_derivatives(problem *prob)
{
    problem_init_jacobian(prob);
    problem_init_hessian(prob);
}

static inline void print_end_message(const Diff_engine_stats *stats)
{
    printf("\n"
           "============================================================\n"
           "              DNLP Differentiation Engine v%s\n"
           "  (c) D. Cederberg and W. Zhang, Stanford University, 2026\n"
           "============================================================\n",
           DIFF_ENGINE_VERSION);

    printf("\nProblem statistics:\n");
    printf("  Affine constraints (nnz):               %d\n", stats->nnz_affine);
    printf("  Jacobian nonlinear constraints (nnz):   %d\n", stats->nnz_nonlinear);
    printf("  Lagrange Hessian (nnz):                 %d\n", stats->nnz_hessian);

    printf("\nTiming (seconds):\n");
    printf("  Derivative structure (sparsity):     %8.3f\n",
           stats->time_init_derivatives);
    printf("  Jacobian evaluation:                 %8.3f\n",
           stats->time_eval_jacobian);
    printf("  Hessian evaluation:                  %8.3f\n",
           stats->time_eval_hessian);
    printf("  Objective evaluation:                %8.3f\n",
           stats->time_forward_obj);
    printf("  Constraints evaluation:              %8.3f\n",
           stats->time_forward_constraints);

    double total_time = stats->time_init_derivatives + stats->time_eval_jacobian +
                        stats->time_eval_hessian + stats->time_forward_obj +
                        stats->time_forward_constraints;

    printf("  ----------------------------------------------\n");
    printf("  Total differentiation time:          %8.3f\n", total_time);
}

void free_problem(problem *prob)
{
    if (prob == NULL) return;

    /* Free allocated arrays */
    free(prob->constraint_values);
    free(prob->gradient_values);
    free_csr_matrix(prob->jacobian);
    free_csr_matrix(prob->lagrange_hessian);
    free(prob->hess_idx_map);

    /* Release expression references (decrements refcount) */
    free_expr(prob->objective);
    for (int i = 0; i < prob->n_constraints; i++)
    {
        free_expr(prob->constraints[i]);
    }
    free(prob->constraints);

    if (prob->verbose)
    {
        print_end_message(&prob->stats);
    }

    /* Free problem struct */
    free(prob);
}

double problem_objective_forward(problem *prob, const double *u)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    /* Evaluate objective only */
    prob->objective->forward(prob->objective, u);
    double result = prob->objective->value[0];

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_forward_obj += GET_ELAPSED_SECONDS(timer);

    return result;
}

void problem_constraint_forward(problem *prob, const double *u)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    /* Evaluate constraints only and copy values */
    int offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->forward(c, u);
        memcpy(prob->constraint_values + offset, c->value, c->size * sizeof(double));
        offset += c->size;
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_forward_constraints += GET_ELAPSED_SECONDS(timer);
}

void problem_gradient(problem *prob)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    /* evaluate jacobian of objective */
    prob->objective->eval_jacobian(prob->objective);

    /* copy sparse jacobian to dense gradient */
    memset(prob->gradient_values, 0, prob->n_vars * sizeof(double));
    CSR_Matrix *jac = prob->objective->jacobian;
    for (int k = jac->p[0]; k < jac->p[1]; k++)
    {
        prob->gradient_values[jac->i[k]] = jac->x[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_gradient += GET_ELAPSED_SECONDS(timer);
}

void problem_jacobian(problem *prob)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    bool first_call = !prob->jacobian_called;

    CSR_Matrix *J = prob->jacobian;
    int nnz_offset = 0;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];

        if (!first_call && c->is_affine(c))
        {
            /* skip evaluation for affine constraints after first call */
            nnz_offset += c->jacobian->nnz;
            continue;
        }

        c->eval_jacobian(c);
        memcpy(J->x + nnz_offset, c->jacobian->x, c->jacobian->nnz * sizeof(double));
        nnz_offset += c->jacobian->nnz;
    }

    /* update actual nnz (may be less than allocated) */
    J->nnz = nnz_offset;

    prob->jacobian_called = true;
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_jacobian += GET_ELAPSED_SECONDS(timer);
}

void problem_hessian(problem *prob, double obj_w, const double *w)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // ------------------------------------------------------------------------
    //             evaluate hessian of objective and constraints
    // ------------------------------------------------------------------------
    expr *obj = prob->objective;
    obj->eval_wsum_hess(obj, &obj_w);

    int offset = 0;
    expr **constrs = prob->constraints;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        if (constrs[i]->is_affine(constrs[i]))
        {
            /* skip evaluation for affine constraints */
            offset += constrs[i]->size;
            continue;
        }
        constrs[i]->eval_wsum_hess(constrs[i], w + offset);
        offset += constrs[i]->size;
    }

    // ------------------------------------------------------------------------
    //           assemble Lagrange hessian using index map
    // ------------------------------------------------------------------------
    CSR_Matrix *H = prob->lagrange_hessian;
    int *idx_map = prob->hess_idx_map;

    /* zero out hessian before adding contribution from obj and constraints */
    memset(H->x, 0, H->nnz * sizeof(double));

    /* accumulate objective function */
    idx_map_accumulator(obj->wsum_hess, idx_map, H->x);
    offset = obj->wsum_hess->nnz;

    /* accumulate constraint functions */
    for (int i = 0; i < prob->n_constraints; i++)
    {
        idx_map_accumulator(constrs[i]->wsum_hess, idx_map + offset, H->x);
        offset += constrs[i]->wsum_hess->nnz;
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_hessian += GET_ELAPSED_SECONDS(timer);
}

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
#include "affine.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* Constants don't depend on u; values are already set */
    (void) node;
    (void) u;
}

static void jacobian_init(expr *node)
{
    /* Constant jacobian is all zeros: size x n_vars with 0 nonzeros.
     * new_csr_matrix uses calloc for row pointers, so they're already 0. */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, 0);
}

static void eval_jacobian(expr *node)
{
    /* Constant jacobian never changes - nothing to evaluate */
    (void) node;
}

static void wsum_hess_init(expr *node)
{
    /* Constant Hessian is all zeros: n_vars x n_vars with 0 nonzeros. */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 0);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Constant Hessian is always zero - nothing to compute */
    (void) node;
    (void) w;
}

static bool is_affine(const expr *node)
{
    (void) node;
    return true;
}

expr *new_constant(int d1, int d2, int n_vars, const double *values)
{
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, d1, d2, n_vars, forward, jacobian_init, eval_jacobian, is_affine,
              wsum_hess_init, eval_wsum_hess, NULL);
    memcpy(node->value, values, node->size * sizeof(double));

    return node;
}

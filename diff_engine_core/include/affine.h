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
#ifndef AFFINE_H
#define AFFINE_H

#include "expr.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"

expr *new_linear(expr *u, const CSR_Matrix *A, const double *b);

expr *new_add(expr *left, expr *right);
expr *new_neg(expr *child);

expr *new_sum(expr *child, int axis);
expr *new_hstack(expr **args, int n_args, int n_vars);
expr *new_promote(expr *child, int d1, int d2);
expr *new_trace(expr *child);

expr *new_constant(int d1, int d2, int n_vars, const double *values);
expr *new_variable(int d1, int d2, int var_id, int n_vars);

expr *new_index(expr *child, int d1, int d2, const int *indices, int n_idxs);
expr *new_reshape(expr *child, int d1, int d2);
expr *new_broadcast(expr *child, int target_d1, int target_d2);
expr *new_diag_vec(expr *child);
expr *new_transpose(expr *child);
expr *new_diag_vec(expr *child);

#endif /* AFFINE_H */

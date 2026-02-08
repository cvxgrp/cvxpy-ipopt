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
#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include "expr.h"

/* Helper function to initialize an elementwise expr (can be used with derived types)
 */
void init_elementwise(expr *node, expr *child);

expr *new_exp(expr *child);
expr *new_log(expr *child);
expr *new_entr(expr *child);
expr *new_sin(expr *child);
expr *new_cos(expr *child);
expr *new_tan(expr *child);
expr *new_sinh(expr *child);
expr *new_tanh(expr *child);
expr *new_asinh(expr *child);
expr *new_atanh(expr *child);
expr *new_logistic(expr *child);
expr *new_power(expr *child, double p);
expr *new_xexp(expr *child);

/* the jacobian and wsum_hess for elementwise univariate atoms are always
   initialized in the same way and implement the chain rule in the same way */
void jacobian_init_elementwise(expr *node);
void eval_jacobian_elementwise(expr *node);
void wsum_hess_init_elementwise(expr *node);
void eval_wsum_hess_elementwise(expr *node, const double *w);
expr *new_elementwise(expr *child);

/* no elementwise atoms are affine according to our convention,
   so we can have a common implementation */
bool is_affine_elementwise(const expr *node);

#endif /* ELEMENTWISE_UNIVARIATE_H */

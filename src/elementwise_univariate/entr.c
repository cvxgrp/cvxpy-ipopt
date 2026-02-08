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
#include "elementwise_univariate.h"
#include <math.h>

static void forward(expr *node, const double *u)
{
    expr *child = node->left;

    /* child's forward pass */
    child->forward(child, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = -child->value[i] * log(child->value[i]);
    }
}

static void local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = -log(child->value[j]) - 1.0;
    }
}

static void local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;

    for (int j = 0; j < node->size; j++)
    {
        out[j] = -w[j] / x[j];
    }
}

expr *new_entr(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;
    return node;
}

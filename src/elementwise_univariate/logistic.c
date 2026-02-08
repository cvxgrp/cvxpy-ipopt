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

    /* child's forward pass */
    node->left->forward(node->left, u);

    double *x = node->left->value;

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        if (x[i] >= 0)
        {
            node->value[i] = x[i] + log(1.0 + exp(-x[i]));
        }
        else
        {
            node->value[i] = log(1.0 + exp(x[i]));
        }
    }
}

static void local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        if (x[j] >= 0)
        {
            vals[j] = 1.0 / (1.0 + exp(-x[j]));
        }
        else
        {
            double exp_x = exp(x[j]);
            vals[j] = exp_x / (1.0 + exp_x);
        }
    }
}

static void local_wsum_hess(expr *node, double *out, const double *w)
{
    double *sigmas;

    if (node->left->var_id != NOT_A_VARIABLE)
    {
        sigmas = node->jacobian->x;
    }
    else
    {
        sigmas = node->dwork;
    }

    for (int j = 0; j < node->size; j++)
    {
        out[j] = sigmas[j] * (1.0 - sigmas[j]) * w[j];
    }
}

expr *new_logistic(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;
    return node;
}

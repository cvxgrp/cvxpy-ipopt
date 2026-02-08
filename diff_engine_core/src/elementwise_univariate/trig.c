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

/* ----------------------- sin ----------------------- */
static void sin_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = sin(node->left->value[i]);
    }
}

static void sin_local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = cos(x[j]);
    }
}

static void sin_local_wsum_hess(expr *node, double *out, const double *w)
{
    for (int j = 0; j < node->size; j++)
    {
        out[j] = -w[j] * node->value[j];
    }
}

expr *new_sin(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = sin_forward;
    node->local_jacobian = sin_local_jacobian;
    node->local_wsum_hess = sin_local_wsum_hess;
    return node;
}

/* ----------------------- cos ----------------------- */
static void cos_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = cos(node->left->value[i]);
    }
}

static void cos_local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = -sin(x[j]);
    }
}

static void cos_local_wsum_hess(expr *node, double *out, const double *w)
{
    for (int j = 0; j < node->size; j++)
    {
        out[j] = -w[j] * node->value[j];
    }
}

expr *new_cos(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = cos_forward;
    node->local_jacobian = cos_local_jacobian;
    node->local_wsum_hess = cos_local_wsum_hess;
    return node;
}

/* ----------------------- tan ----------------------- */
static void tan_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = tan(node->left->value[i]);
    }
}

static void tan_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        double c = cos(child->value[j]);
        vals[j] = 1.0 / (c * c);
    }
}

static void tan_local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;

    for (int j = 0; j < node->size; j++)
    {
        double c = cos(x[j]);
        out[j] = 2.0 * w[j] * node->value[j] / (c * c);
    }
}

expr *new_tan(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = tan_forward;
    node->local_jacobian = tan_local_jacobian;
    node->local_wsum_hess = tan_local_wsum_hess;
    return node;
}

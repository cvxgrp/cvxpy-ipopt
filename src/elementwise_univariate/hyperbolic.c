#include "elementwise_univariate.h"
#include <assert.h>
#include <math.h>

/* ----------------------- sinh ----------------------- */
static void sinh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = sinh(node->left->value[i]);
    }
}

static void sinh_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = cosh(child->value[j]);
    }
}

static void sinh_local_wsum_hess(expr *node, double *out, const double *w)
{
    for (int j = 0; j < node->size; j++)
    {
        out[j] = w[j] * node->value[j];
    }
}

expr *new_sinh(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = sinh_forward;
    node->local_jacobian = sinh_local_jacobian;
    node->local_wsum_hess = sinh_local_wsum_hess;
    return node;
}

/* ----------------------- tanh ----------------------- */
static void tanh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = tanh(node->left->value[i]);
    }
}

static void tanh_local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = cosh(x[j]);
        vals[j] = 1.0 / (c * c);
    }
}

static void tanh_local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = cosh(x[j]);
        out[j] = w[j] * (-2.0 * node->value[j] / (c * c));
    }
}

expr *new_tanh(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = tanh_forward;
    node->local_jacobian = tanh_local_jacobian;
    node->local_wsum_hess = tanh_local_wsum_hess;
    return node;
}

/* ----------------------- asinh ----------------------- */
static void asinh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = asinh(node->left->value[i]);
    }
}

static void asinh_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = 1.0 / sqrt(1.0 + child->value[j] * child->value[j]);
    }
}

static void asinh_local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = 1.0 + x[j] * x[j];
        out[j] = w[j] * (-x[j]) / pow(c, 1.5);
    }
}

expr *new_asinh(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = asinh_forward;
    node->local_jacobian = asinh_local_jacobian;
    node->local_wsum_hess = asinh_local_wsum_hess;
    return node;
}

/* ----------------------- atanh ----------------------- */
static void atanh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = atanh(node->left->value[i]);
    }
}

static void atanh_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = 1.0 / (1.0 - child->value[j] * child->value[j]);
    }
}

static void atanh_local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = 1.0 - x[j] * x[j];
        out[j] = w[j] * (2.0 * x[j]) / (c * c);
    }
}

expr *new_atanh(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = atanh_forward;
    node->local_jacobian = atanh_local_jacobian;
    node->local_wsum_hess = atanh_local_wsum_hess;
    return node;
}

#include "other.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IS_ZERO(x) (fabs((x)) < 1e-8)

static inline void wsum_hess_no_zeros(expr *node, const double *w);
static inline void wsum_hess_one_zero(expr *node, const double *w);
static inline void wsum_hess_two_zeros(expr *node, const double *w);
static inline void wsum_hess_many_zeros(expr *node, const double *w);

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* forward pass of child */
    x->forward(x, u);

    /* local forward pass and count zeros */
    double prod_nonzero = 1.0;
    int zeros = 0;
    int zero_idx = -1;
    for (int i = 0; i < x->size; i++)
    {
        if (IS_ZERO(x->value[i]))
        {
            zeros++;
            zero_idx = i;
        }
        else
        {
            prod_nonzero *= x->value[i];
        }
    }

    node->value[0] = (zeros > 0) ? 0.0 : prod_nonzero;
    prod_expr *pnode = (prod_expr *) node;
    pnode->num_of_zeros = zeros;
    pnode->zero_index = zero_idx;
    pnode->prod_nonzero = prod_nonzero;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(1, node->n_vars, x->size);
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = x->size;
        for (int j = 0; j < x->size; j++)
        {
            node->jacobian->i[j] = x->var_id + j;
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod jacobian_init: non-variable child not implemented\n");
        exit(1);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    prod_expr *pnode = (prod_expr *) node;
    int num_of_zeros = pnode->num_of_zeros;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        if (num_of_zeros == 0)
        {
            for (int j = 0; j < x->size; j++)
            {
                node->jacobian->x[j] = node->value[0] / x->value[j];
            }
        }
        else if (num_of_zeros == 1)
        {
            memset(node->jacobian->x, 0, sizeof(double) * x->size);
            node->jacobian->x[pnode->zero_index] = pnode->prod_nonzero;
        }
        else
        {
            memset(node->jacobian->x, 0, sizeof(double) * x->size);
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod eval_jacobian: non-variable child not implemented\n");
        exit(1);
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* allocate n_vars x n_vars CSR matrix with dense block */
        int block_size = x->size;
        int nnz = block_size * block_size;
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz);

        /* fill row pointers for the dense block */
        for (int i = 0; i < block_size; i++)
        {
            node->wsum_hess->p[x->var_id + i] = i * block_size;
        }

        /* fill row pointers for rows after the block */
        for (int i = x->var_id + block_size; i <= node->n_vars; i++)
        {
            node->wsum_hess->p[i] = nnz;
        }

        /* fill column indices for the dense block */
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                node->wsum_hess->i[i * block_size + j] = x->var_id + j;
            }
        }
    }
    else
    {
        fprintf(
            stderr,
            "Error in prod wsum_hess_init: non-variable child not implemented\n");
        exit(1);
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    int num_of_zeros = ((prod_expr *) node)->num_of_zeros;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        if (num_of_zeros == 0)
        {
            wsum_hess_no_zeros(node, w);
        }
        else if (num_of_zeros == 1)
        {
            wsum_hess_one_zero(node, w);
        }
        else if (num_of_zeros == 2)
        {
            wsum_hess_two_zeros(node, w);
        }
        else
        {
            wsum_hess_many_zeros(node, w);
        }
    }
    else
    {
        fprintf(
            stderr,
            "Error in prod eval_wsum_hess: non-variable child not implemented\n");
        exit(1);
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

static void free_type_data(expr *node)
{
    (void) node;
}

/* when we implement axis-support, check convention of numpy and cvxpy.
I think they return row vectors.*/
expr *new_prod(expr *child)
{
    /* Output is scalar: 1 x 1 */
    prod_expr *pnode = (prod_expr *) calloc(1, sizeof(prod_expr));
    expr *node = &pnode->base;
    init_expr(node, 1, 1, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = child;
    expr_retain(child);
    return node;
}

// ---------------------------------------------------------------------------------------
//                   Helper functions for Hessian evaluation
// ---------------------------------------------------------------------------------------
static inline void wsum_hess_no_zeros(expr *node, const double *w)
{
    double *x = node->left->value;
    int n = node->left->size;
    double wf = w[0] * node->value[0];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                node->wsum_hess->x[i * n + j] = 0.0;
            }
            else
            {
                node->wsum_hess->x[i * n + j] = wf / (x[i] * x[j]);
            }
        }
    }
}

static inline void wsum_hess_one_zero(expr *node, const double *w)
{
    expr *x = node->left;
    double *H = node->wsum_hess->x;
    memset(H, 0, sizeof(double) * (x->size * x->size));
    int p = ((prod_expr *) node)->zero_index;
    double prod_nonzero = ((prod_expr *) node)->prod_nonzero;
    double w_prod = w[0] * prod_nonzero;

    /* fill row p and column p */
    for (int j = 0; j < x->size; j++)
    {
        if (j == p) continue;

        double hess_val = w_prod / x->value[j];
        H[p * x->size + j] = hess_val;
        H[j * x->size + p] = hess_val;
    }
}

static inline void wsum_hess_two_zeros(expr *node, const double *w)
{
    expr *x = node->left;
    int n = x->size;
    memset(node->wsum_hess->x, 0, sizeof(double) * (n * n));

    /* find indices p and q where x[p] = x[q] = 0 */
    int p = -1, q = -1;
    for (int i = 0; i < n; i++)
    {
        if (IS_ZERO(x->value[i]))
        {
            if (p == -1)
            {
                p = i;
            }
            else
            {
                q = i;
                break;
            }
        }
    }
    assert(p != -1 && q != -1);

    double hess_val = w[0] * ((prod_expr *) node)->prod_nonzero;
    node->wsum_hess->x[p * n + q] = hess_val;
    node->wsum_hess->x[q * n + p] = hess_val;
}

static inline void wsum_hess_many_zeros(expr *node, const double *w)
{
    expr *x = node->left;
    memset(node->wsum_hess->x, 0, sizeof(double) * (x->size * x->size));
    (void) w;
}

#include "other.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IS_ZERO(x) (fabs((x)) < 1e-8)

static void forward(expr *node, const double *u)
{
    /* forward pass of child */
    expr *x = node->left;
    x->forward(x, u);

    prod_axis *pnode = (prod_axis *) node;
    int d1 = x->d1;
    int d2 = x->d2;

    /* iterate over columns and compute columnwise products */
    for (int col = 0; col < d2; col++)
    {
        double prod_nonzero = 1.0;
        int zeros = 0;
        int zero_row = -1;
        int start = col * d1;
        int end = start + d1;

        /* iterate over rows of this column */
        for (int idx = start; idx < end; idx++)
        {
            if (IS_ZERO(x->value[idx]))
            {
                zeros++;
                zero_row = idx - start;
            }
            else
            {
                prod_nonzero *= x->value[idx];
            }
        }

        /* store results for this column */
        pnode->num_of_zeros[col] = zeros;
        pnode->zero_index[col] = zero_row;
        pnode->prod_nonzero[col] = prod_nonzero;
        node->value[col] = (zeros > 0) ? 0.0 : prod_nonzero;
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(node->size, node->n_vars, x->size);

        /* set row pointers (each row has d1 nnzs) */
        for (int row = 0; row < x->d2; row++)
        {
            node->jacobian->p[row] = row * x->d1;
        }
        node->jacobian->p[x->d2] = x->size;

        /* set column indices */
        for (int col = 0; col < x->d2; col++)
        {
            int start = col * x->d1;
            for (int i = 0; i < x->d1; i++)
            {
                node->jacobian->i[start + i] = x->var_id + start + i;
            }
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod_axis_zero jacobian_init: child must be a variable\n");
        exit(1);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;

    double *J_vals = node->jacobian->x;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* process each column */
        for (int col = 0; col < x->d2; col++)
        {
            int num_zeros = pnode->num_of_zeros[col];
            int start = col * x->d1;

            if (num_zeros == 0)
            {
                for (int i = 0; i < x->d1; i++)
                {
                    J_vals[start + i] = node->value[col] / x->value[start + i];
                }
            }
            else if (num_zeros == 1)
            {
                memset(J_vals + start, 0, sizeof(double) * x->d1);
                J_vals[start + pnode->zero_index[col]] = pnode->prod_nonzero[col];
            }
            else
            {
                memset(J_vals + start, 0, sizeof(double) * x->d1);
            }
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod_axis_zero eval_jacobian: child must be a variable\n");
        exit(1);
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* Hessian has block diagonal structure: d2 blocks of size d1 x d1 */
        int nnz = x->d2 * x->d1 * x->d1;
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz);
        CSR_Matrix *H = node->wsum_hess;

        /* fill row pointers for the variable's rows (block diagonal) */
        for (int i = 0; i < x->size; i++)
        {
            int block_idx = i / x->d1;
            int row_in_block = i % x->d1;
            H->p[x->var_id + i] = block_idx * x->d1 * x->d1 + row_in_block * x->d1;
        }

        /* fill row pointers for rows after the variable */
        for (int i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            H->p[i] = nnz;
        }

        /* fill column indices for the d2 diagonal blocks */
        for (int block = 0; block < x->d2; block++)
        {
            int block_start_col = x->var_id + block * x->d1;
            for (int i = 0; i < x->d1; i++)
            {
                for (int j = 0; j < x->d1; j++)
                {
                    int idx = block * x->d1 * x->d1 + i * x->d1 + j;
                    H->i[idx] = block_start_col + j;
                }
            }
        }
    }
    else
    {
        fprintf(
            stderr,
            "Error in prod_axis_zero wsum_hess_init: child must be a variable\n");
        exit(1);
    }
}

static inline void wsum_hess_column_no_zeros(expr *node, const double *w, int col,
                                             int d1)
{
    expr *x = node->left;
    double *H = node->wsum_hess->x;
    int col_start = col * d1;
    int block_start = col * d1 * d1;
    double scale = w[col] * node->value[col];

    for (int i = 0; i < d1; i++)
    {
        for (int j = 0; j < d1; j++)
        {
            if (i == j)
            {
                H[block_start + i * d1 + j] = 0.0;
            }
            else
            {
                int idx_i = col_start + i;
                int idx_j = col_start + j;
                H[block_start + i * d1 + j] =
                    scale / (x->value[idx_i] * x->value[idx_j]);
            }
        }
    }
}

static inline void wsum_hess_column_one_zero(expr *node, const double *w, int col,
                                             int d1)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;
    double *H = node->wsum_hess->x;
    int col_start = col * d1;
    int block_start = col * d1 * d1;

    /* clear this column's block */
    memset(&H[block_start], 0, sizeof(double) * d1 * d1);

    int p = pnode->zero_index[col];
    double prod_nonzero = pnode->prod_nonzero[col];
    double w_prod = w[col] * prod_nonzero;

    /* fill row p and column p of this block */
    for (int j = 0; j < d1; j++)
    {
        if (j == p) continue;

        int idx_j = col_start + j;
        double hess_val = w_prod / x->value[idx_j];
        H[block_start + p * d1 + j] = hess_val;
        H[block_start + j * d1 + p] = hess_val;
    }
}

static inline void wsum_hess_column_two_zeros(expr *node, const double *w, int col,
                                              int d1)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;
    double *H = node->wsum_hess->x;
    int col_start = col * d1;
    int block_start = col * d1 * d1;

    /* clear this column's block */
    memset(&H[block_start], 0, sizeof(double) * d1 * d1);

    /* find indices p and q where x[col_start + p] and x[col_start + q] are zero */
    int p = -1, q = -1;
    for (int i = 0; i < d1; i++)
    {
        if (IS_ZERO(x->value[col_start + i]))
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

    double hess_val = w[col] * pnode->prod_nonzero[col];
    H[block_start + p * d1 + q] = hess_val;
    H[block_start + q * d1 + p] = hess_val;
}

static inline void wsum_hess_column_many_zeros(expr *node, const double *w, int col,
                                               int d1)
{
    double *H = node->wsum_hess->x;
    int block_start = col * d1 * d1;

    /* clear this column's block */
    memset(&H[block_start], 0, sizeof(double) * d1 * d1);
    (void) w;
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;
    int d1 = x->d1;
    int d2 = x->d2;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* process each column */
        for (int col = 0; col < d2; col++)
        {
            int num_zeros = pnode->num_of_zeros[col];

            if (num_zeros == 0)
            {
                wsum_hess_column_no_zeros(node, w, col, d1);
            }
            else if (num_zeros == 1)
            {
                wsum_hess_column_one_zero(node, w, col, d1);
            }
            else if (num_zeros == 2)
            {
                wsum_hess_column_two_zeros(node, w, col, d1);
            }
            else
            {
                wsum_hess_column_many_zeros(node, w, col, d1);
            }
        }
    }
    else
    {
        fprintf(
            stderr,
            "Error in prod_axis_zero eval_wsum_hess: child must be a variable\n");
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
    prod_axis *pnode = (prod_axis *) node;
    free(pnode->num_of_zeros);
    free(pnode->zero_index);
    free(pnode->prod_nonzero);
}

/* TODO: refactor to remove diagonal entry as nonzero since it's always zero */
expr *new_prod_axis_zero(expr *child)
{
    prod_axis *pnode = (prod_axis *) calloc(1, sizeof(prod_axis));
    expr *node = &pnode->base;

    /* output is always a row vector 1 x d2 - TODO: is that correct? */
    init_expr(node, 1, child->d2, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess,
              free_type_data);

    /* allocate arrays to store per-column statistics */
    pnode->num_of_zeros = (int *) calloc(child->d2, sizeof(int));
    pnode->zero_index = (int *) calloc(child->d2, sizeof(int));
    pnode->prod_nonzero = (double *) calloc(child->d2, sizeof(double));

    node->left = child;
    expr_retain(child);

    return node;
}

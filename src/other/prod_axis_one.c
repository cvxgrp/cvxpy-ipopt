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

    /* initialize per-row statistics */
    for (int i = 0; i < d1; i++)
    {
        pnode->num_of_zeros[i] = 0;
        pnode->zero_index[i] = -1;
        pnode->prod_nonzero[i] = 1.0;
    }

    /* iterate over columns */
    for (int col = 0; col < d2; col++)
    {
        int start = col * d1;
        int end = start + d1;

        for (int idx = start; idx < end; idx++)
        {
            int row = idx - start;
            if (IS_ZERO(x->value[idx]))
            {
                pnode->num_of_zeros[row]++;
                pnode->zero_index[row] = col;
            }
            else
            {
                pnode->prod_nonzero[row] *= x->value[idx];
            }
        }
    }

    /* compute output values */
    for (int i = 0; i < d1; i++)
    {
        node->value[i] = (pnode->num_of_zeros[i] > 0) ? 0.0 : pnode->prod_nonzero[i];
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

        /* set row pointers (each row has d2 nnzs) */
        for (int row = 0; row < x->d1; row++)
        {
            node->jacobian->p[row] = row * x->d2;
        }
        node->jacobian->p[x->d1] = x->size;

        /* set column indices */
        for (int row = 0; row < x->d1; row++)
        {
            int start = row * x->d2;
            for (int col = 0; col < x->d2; col++)
            {
                node->jacobian->i[start + col] = x->var_id + col * x->d1 + row;
            }
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod_axis_one jacobian_init: child must be a variable\n");
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
        /* process each row */
        for (int row = 0; row < x->d1; row++)
        {
            int num_zeros = pnode->num_of_zeros[row];
            int start = row * x->d2;

            if (num_zeros == 0)
            {
                for (int col = 0; col < x->d2; col++)
                {
                    int idx = col * x->d1 + row;
                    J_vals[start + col] = node->value[row] / x->value[idx];
                }
            }
            else if (num_zeros == 1)
            {
                memset(J_vals + start, 0, sizeof(double) * x->d2);
                J_vals[start + pnode->zero_index[row]] = pnode->prod_nonzero[row];
            }
            else
            {
                memset(J_vals + start, 0, sizeof(double) * x->d2);
            }
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod_axis_one eval_jacobian: child must be a variable\n");
        exit(1);
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* each row i has d2-1 non-zero entries, with column indices corresponding to
           the columns in that row (except the diagonal element). */
        int nnz = x->d1 * x->d2 * (x->d2 - 1);
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz);
        CSR_Matrix *H = node->wsum_hess;

        /* fill sparsity pattern */
        int nnz_per_row = x->d2 - 1;
        for (int i = 0; i < x->size; i++)
        {
            int row = i % x->d1;
            int col = i / x->d1;
            int start = i * nnz_per_row;
            H->p[x->var_id + i] = start;

            /* for this variable (at X[row, col]), the Hessian has entries with all
             * columns in the same row, excluding the diagonal (col itself) */
            int offset = 0;
            int col_offset = x->var_id + row;
            for (int j = 0; j < x->d2; j++)
            {
                if (j != col)
                {
                    H->i[start + offset] = col_offset + j * x->d1;
                    offset++;
                }
            }
        }

        /* fill row pointers for rows after the variable */
        for (int i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            H->p[i] = nnz;
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod_axis_one wsum_hess_init: child must be a variable\n");
        exit(1);
    }
}

static inline void wsum_hess_row_no_zeros(expr *node, const double *w, int row,
                                          int d2)
{
    expr *x = node->left;
    CSR_Matrix *H = node->wsum_hess;
    double scale = w[row] * node->value[row];

    /* for each variable xk in this row, fill in Hessian entries
    //    TODO: more cache-friendly solution is possible? */
    for (int i = 0; i < d2; i++)
    {
        int k = x->var_id + row + i * x->d1;
        int idx_k = i * x->d1 + row;
        int offset = 0;
        for (int j = 0; j < d2; j++)
        {
            if (j != i)
            {
                int idx_j = j * x->d1 + row;
                H->x[H->p[k] + offset] = scale / (x->value[idx_k] * x->value[idx_j]);
                offset++;
            }
        }
    }
}

static inline void wsum_hess_row_one_zero(expr *node, const double *w, int row,
                                          int d2)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;
    int p = pnode->zero_index[row]; /* zero column index */
    double w_prod = w[row] * pnode->prod_nonzero[row];

    /* For each variable in this row */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        int row_start = H->p[var_i];

        /* Row var_i has d2-1 entries (excluding diagonal) */
        int offset = 0;
        for (int j = 0; j < d2; j++)
        {
            if (j != i)
            {
                if (i == p && j != p)
                {
                    /* row p (zero row), column j (nonzero) */
                    int idx_j = j * x->d1 + row;
                    H_vals[row_start + offset] = w_prod / x->value[idx_j];
                }
                else if (j == p && i != p)
                {
                    /* row i (nonzero), column p (zero) */
                    int idx_i = i * x->d1 + row;
                    H_vals[row_start + offset] = w_prod / x->value[idx_i];
                }
                else
                {
                    /* all other entries are zero */
                    H_vals[row_start + offset] = 0.0;
                }
                offset++;
            }
        }
    }
}

static inline void wsum_hess_row_two_zeros(expr *node, const double *w, int row,
                                           int d2)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;

    /* find indices p and q where row has zeros */
    int p = -1, q = -1;
    for (int c = 0; c < d2; c++)
    {
        if (IS_ZERO(x->value[c * node->left->d1 + row]))
        {
            if (p == -1)
            {
                p = c;
            }
            else
            {
                q = c;
                break;
            }
        }
    }
    assert(p != -1 && q != -1);

    double hess_val = w[row] * pnode->prod_nonzero[row];

    /* For each variable in this row */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        int row_start = H->p[var_i];

        /* row var_i has d2-1 entries (excluding diagonal) */
        int offset = 0;
        for (int j = 0; j < d2; j++)
        {
            if (j != i)
            {
                /* only (p,q) and (q,p) are nonzero */
                if ((i == p && j == q) || (i == q && j == p))
                {
                    H_vals[row_start + offset] = hess_val;
                }
                else
                {
                    H_vals[row_start + offset] = 0.0;
                }
                offset++;
            }
        }
    }
}

static inline void wsum_hess_row_many_zeros(expr *node, int row, int d2)
{
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;
    expr *x = node->left;

    /* for each variable in this row, zero out all entries */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        memset(H_vals + H->p[var_i], 0, sizeof(double) * (d2 - 1));
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    prod_axis *pnode = (prod_axis *) node;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        for (int row = 0; row < x->d1; row++)
        {
            int num_zeros = pnode->num_of_zeros[row];

            if (num_zeros == 0)
            {
                wsum_hess_row_no_zeros(node, w, row, x->d2);
            }
            else if (num_zeros == 1)
            {
                wsum_hess_row_one_zero(node, w, row, x->d2);
            }
            else if (num_zeros == 2)
            {
                wsum_hess_row_two_zeros(node, w, row, x->d2);
            }
            else
            {
                wsum_hess_row_many_zeros(node, row, x->d2);
            }
        }
    }
    else
    {
        fprintf(stderr,
                "Error in prod_axis_one eval_wsum_hess: child must be a variable\n");
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

expr *new_prod_axis_one(expr *child)
{
    prod_axis *pnode = (prod_axis *) calloc(1, sizeof(prod_axis));
    expr *node = &pnode->base;

    /* output is always a row vector 1 x d1 (one product per row) */
    init_expr(node, 1, child->d1, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess,
              free_type_data);

    /* allocate arrays to store per-row statistics */
    pnode->num_of_zeros = (int *) calloc(child->d1, sizeof(int));
    pnode->zero_index = (int *) calloc(child->d1, sizeof(int));
    pnode->prod_nonzero = (double *) calloc(child->d1, sizeof(double));

    node->left = child;
    expr_retain(child);

    return node;
}

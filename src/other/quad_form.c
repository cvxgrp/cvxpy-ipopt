#include "other.h"
#include "subexpr.h"
#include "utils/CSC_Matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    x->forward(x, u);

    /* local forward pass  */
    CSR_Matrix *Q = ((quad_form_expr *) node)->Q;
    csr_matvec(Q, x->value, node->dwork, 0);
    node->value[0] = 0.0;

    for (int i = 0; i < x->size; i++)
    {
        node->value[0] += x->value[i] * node->dwork[i];
    }
}

static void jacobian_init(expr *node)
{
    assert(node->left->var_id != NOT_A_VARIABLE);

    expr *x = node->left;
    node->dwork = (double *) malloc(x->size * sizeof(double));
    node->jacobian = new_csr_matrix(1, node->n_vars, x->size);
    node->jacobian->p[0] = 0;
    node->jacobian->p[1] = x->size;

    for (int j = 0; j < x->size; j++)
    {
        node->jacobian->i[j] = x->var_id + j;
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    CSR_Matrix *Q = ((quad_form_expr *) node)->Q;

    // jacobian = 2 * Q * x
    csr_matvec(Q, x->value, node->jacobian->x, 0);

    for (int j = 0; j < x->size; j++)
    {
        node->jacobian->x[j] *= 2.0;
    }
}

static void wsum_hess_init(expr *node)
{
    CSR_Matrix *Q = ((quad_form_expr *) node)->Q;
    expr *x = node->left;
    CSR_Matrix *H = new_csr_matrix(node->n_vars, node->n_vars, Q->nnz);

    /* set global row pointers */
    memcpy(H->p + x->var_id, Q->p, (x->size + 1) * sizeof(int));
    for (int i = x->var_id + x->size + 1; i <= node->n_vars; i++)
    {

        H->p[i] = Q->nnz;
    }

    /* set global column indices */
    for (int i = 0; i < Q->nnz; i++)
    {
        H->i[i] = Q->i[i] + x->var_id;
    }

    node->wsum_hess = H;
}

static void eval_wsum_hess(expr *node, const double *w)
{
    CSR_Matrix *Q = ((quad_form_expr *) node)->Q;
    double *H = node->wsum_hess->x;
    double two_w = 2.0 * w[0];
    for (int i = 0; i < Q->nnz; i++)
    {
        H[i] = two_w * Q->x[i];
    }
}

/*
The following two functions are commented out. It supports the jacobian for
quad_form(Ax, Q), but after reconsideration, I think we should treat this as
quad_form(x, A.TQA) in the canonicalization so we don't need to support the chain
rule here.
static void jacobian_init(expr *node)
{
    expr *x = node->left;
    node->dwork = (double *) malloc(x->d1 * sizeof(double));

    // if x is a variable
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(1, node->n_vars, x->d1);
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = x->d1;

        for (int j = 0; j < x->d1; j++)
        {
            node->jacobian->i[j] = x->var_id + j;
        }
    }
    else // x is not a variable
    {
        // compute required allocation and allocate jacobian
        bool *col_nz = (bool *) calloc(node->n_vars, sizeof(bool));
        int nonzero_cols = count_nonzero_cols(x->jacobian, col_nz);
        node->jacobian = new_csr_matrix(1, node->n_vars, nonzero_cols + 1);

        // precompute column indices
        node->jacobian->nnz = 0;
        for (int j = 0; j < node->n_vars; j++)
        {
            if (col_nz[j])
            {
                node->jacobian->i[node->jacobian->nnz] = j;
                node->jacobian->nnz++;
            }
        }
        assert(nonzero_cols == node->jacobian->nnz);
        free(col_nz);

        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = node->jacobian->nnz;
    }
}


static void eval_jacobian_old(expr *node)
{
    expr *x = node->left;
    CSR_Matrix *Q = ((quad_form_expr *) node)->Q;

    // if x is a variable
    if (x->var_id != NOT_A_VARIABLE)
    {
        csr_matvec(Q, x->value, node->jacobian->x, 0);

        for (int j = 0; j < x->d1; j++)
        {
            node->jacobian->x[j] *= 2.0;
        }
    }
    else // x is not a variable
    {
        linear_op_expr *lin_x = (linear_op_expr *) x;

        // local jacobian
        csr_matvec(Q, x->value, node->dwork, 0);

        for (int j = 0; j < x->d1; j++)
        {
            node->dwork[j] *= 2.0;
        }

        // chain rule using CSC format
        csc_matvec_fill_values(lin_x->A_csc, node->dwork, node->jacobian);
    }
}
*/

static void free_type_data(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    free_csr_matrix(qnode->Q);
    qnode->Q = NULL;
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_quad_form(expr *left, CSR_Matrix *Q)
{
    assert(left->d1 == 1 || left->d2 == 1); /* left must be a vector */
    quad_form_expr *qnode = (quad_form_expr *) calloc(1, sizeof(quad_form_expr));
    expr *node = &qnode->base;

    init_expr(node, 1, 1, left->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = left;
    expr_retain(left);

    /* Set type-specific field */
    qnode->Q = new_csr_matrix(Q->m, Q->n, Q->nnz);
    copy_csr_matrix(Q, qnode->Q);
    return node;
}

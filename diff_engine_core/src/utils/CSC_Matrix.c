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
#include "utils/CSC_Matrix.h"
#include "utils/iVec.h"
#include <stdlib.h>
#include <string.h>

CSC_Matrix *new_csc_matrix(int m, int n, int nnz)
{
    CSC_Matrix *matrix = (CSC_Matrix *) malloc(sizeof(CSC_Matrix));
    if (!matrix) return NULL;

    matrix->p = (int *) malloc((n + 1) * sizeof(int));
    matrix->i = (int *) malloc(nnz * sizeof(int));
    matrix->x = (double *) malloc(nnz * sizeof(double));

    if (!matrix->p || !matrix->i || !matrix->x)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
        return NULL;
    }

    matrix->m = m;
    matrix->n = n;
    matrix->nnz = nnz;

    return matrix;
}

void free_csc_matrix(CSC_Matrix *matrix)
{
    if (matrix)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
    }
}

CSR_Matrix *ATA_alloc(const CSC_Matrix *A)
{
    /* A is m x n, A^T A is n x n */
    int n = A->n;
    int m = A->m;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for upper triangular part of C = A^T A */
    int *Cp = (int *) malloc((n + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);
    Cp[0] = 0;

    /* compute sparsity pattern, only storing upper triangular part */
    for (i = 0; i < n; i++)
    {
        /* check if Cij != 0 */
        for (j = i; j < n; j++)
        {
            ii = A->p[i];
            jj = A->p[j];

            while (ii < A->p[i + 1] && jj < A->p[j + 1])
            {
                if (A->i[ii] == A->i[jj])
                {
                    nnz += (j != i) ? 2 : 1;
                    iVec_append(Ci, j);
                    break;
                }
                else if (A->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C and symmetrize it */
    CSR_Matrix *C = new_csr_matrix(n, n, nnz);
    symmetrize_csr(Cp, Ci->data, n, C);

    /* free workspace */
    free(Cp);
    iVec_free(Ci);

    return C;
}

static inline double sparse_wdot(const double *a_x, const int *a_i, int a_nnz,
                                 const double *b_x, const int *b_i, int b_nnz,
                                 const double *d)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj])
        {
            sum += a_x[ii] * b_x[jj] * d[a_i[ii]];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj])
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

/* Unweighted sparse dot product of two sorted index arrays */
static inline double sparse_dot(const double *a_x, const int *a_i, int a_nnz,
                                const double *b_x, const int *b_i, int b_nnz)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj])
        {
            sum += a_x[ii] * b_x[jj];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj])
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

void ATDA_fill_values(const CSC_Matrix *A, const double *d, CSR_Matrix *C)
{
    int j, ii, jj;
    for (ii = 0; ii < C->m; ii++)
    {
        for (jj = C->p[ii]; jj < C->p[ii + 1]; jj++)
        {
            j = C->i[jj];

            if (j < ii)
            {
                C->x[jj] = csr_get_value(C, j, ii);
            }
            else
            {
                int nnz_ai = A->p[ii + 1] - A->p[ii];
                int nnz_aj = A->p[j + 1] - A->p[j];

                /* compute Cij = weighted inner product of column i and column j */
                double sum = sparse_wdot(A->x + A->p[ii], A->i + A->p[ii], nnz_ai,
                                         A->x + A->p[j], A->i + A->p[j], nnz_aj, d);

                C->x[jj] = sum;
            }
        }
    }
}

CSC_Matrix *csr_to_csc(const CSR_Matrix *A)
{
    CSC_Matrix *C = new_csc_matrix(A->m, A->n, A->nnz);

    int i, j;
    int *count = malloc(A->n * sizeof(int));

    memset(count, 0, A->n * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each column of A
    // -------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute column pointers
    // ------------------------------------------------------------------
    C->p[0] = 0;
    for (i = 0; i < A->n; ++i)
    {
        C->p[i + 1] = C->p[i] + count[i];
        count[i] = C->p[i];
    }

    // ------------------------------------------------------------------
    //                         fill matrix
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->x[count[A->i[j]]] = A->x[j];
            C->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    free(count);
    return C;
}

CSC_Matrix *csr_to_csc_fill_sparsity(const CSR_Matrix *A, int *iwork)
{
    CSC_Matrix *C = new_csc_matrix(A->m, A->n, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->n * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each column of A
    // -------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute column pointers
    // ------------------------------------------------------------------
    C->p[0] = 0;
    for (i = 0; i < A->n; ++i)
    {
        C->p[i + 1] = C->p[i] + count[i];
        count[i] = C->p[i];
    }

    // ------------------------------------------------------------------
    //                         fill row indices
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return C;
}

void csr_to_csc_fill_values(const CSR_Matrix *A, CSC_Matrix *C, int *iwork)
{
    int i, j;
    int *count = iwork;
    memcpy(count, C->p, A->n * sizeof(int));

    // ------------------------------------------------------------------
    //                         fill values
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->x[count[A->i[j]]] = A->x[j];
            count[A->i[j]]++;
        }
    }
}

CSR_Matrix *BTA_alloc(const CSC_Matrix *A, const CSC_Matrix *B)
{
    /* A is m x n, B is m x p, C = B^T A is p x n */
    int n = A->n;
    int p = B->n;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for C = B^T A */
    int *Cp = (int *) malloc((p + 1) * sizeof(int));
    iVec *Ci = iVec_new(n);
    Cp[0] = 0;

    /* compute sparsity pattern */
    for (i = 0; i < p; i++)
    {
        /* check if Cij != 0 for each column j of A */
        for (j = 0; j < n; j++)
        {
            ii = B->p[i];
            jj = A->p[j];

            /* check if row i of B^T (column i of B) has common row with column j of
             * A */
            while (ii < B->p[i + 1] && jj < A->p[j + 1])
            {
                if (B->i[ii] == A->i[jj])
                {
                    nnz++;
                    iVec_append(Ci, j);
                    break;
                }
                else if (B->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C */
    CSR_Matrix *C = new_csr_matrix(p, n, nnz);
    memcpy(C->p, Cp, (p + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));

    /* free workspace */
    free(Cp);
    iVec_free(Ci);

    return C;
}

void csc_matvec_fill_values(const CSC_Matrix *A, const double *z, CSR_Matrix *C)
{
    /* Compute C = z^T * A where A is in CSC format
     * C is a single-row CSR matrix with column indices pre-computed
     * This fills in the values of C only
     */

    for (int col = 0; col < A->n; col++)
    {
        double val = 0;
        for (int j = A->p[col]; j < A->p[col + 1]; j++)
        {
            val += z[A->i[j]] * A->x[j];
        }

        if (A->p[col + 1] - A->p[col] == 0) continue;

        /* find position in C and fill value */
        for (int k = 0; k < C->nnz; k++)
        {
            if (C->i[k] == col)
            {
                C->x[k] = val;
                break;
            }
        }
    }
}

void BTDA_fill_values(const CSC_Matrix *A, const CSC_Matrix *B, const double *d,
                      CSR_Matrix *C)
{
    int i, j, jj;
    for (i = 0; i < C->m; i++)
    {
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];

            int nnz_bi = B->p[i + 1] - B->p[i];
            int nnz_aj = A->p[j + 1] - A->p[j];

            /* compute Cij = weighted inner product of col i of B and col j of A */
            double sum = sparse_wdot(B->x + B->p[i], B->i + B->p[i], nnz_bi,
                                     A->x + A->p[j], A->i + A->p[j], nnz_aj, d);

            C->x[jj] = sum;
        }
    }
}

static bool has_overlap(const int *a_idx, int a_len, const int *b_idx, int b_len)
{
    int ai = 0, bi = 0;
    while (ai < a_len && bi < b_len)
    {
        if (a_idx[ai] == b_idx[bi]) return true;
        if (a_idx[ai] < b_idx[bi])
        {
            ai++;
        }
        else
        {
            bi++;
        }
    }
    return false;
}

/* Fill values of C = A @ B where A is CSR, B is CSC. */
void csr_csc_matmul_fill_values(const CSR_Matrix *A, const CSC_Matrix *B,
                                CSR_Matrix *C)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            int j = C->i[jj];

            int a_nnz = A->p[i + 1] - A->p[i];
            int b_nnz = B->p[j + 1] - B->p[j];

            /* Compute dot product of row i of A and column j of B */
            double sum = sparse_dot(A->x + A->p[i], A->i + A->p[i], a_nnz,
                                    B->x + B->p[j], B->i + B->p[j], b_nnz);

            C->x[jj] = sum;
        }
    }
}

/* C = A @ B where A is CSR (m x n), B is CSC (n x p). Result C is CSR (m x p)
  with precomputed sparsity pattern */
CSR_Matrix *csr_csc_matmul_alloc(const CSR_Matrix *A, const CSC_Matrix *B)
{
    int m = A->m;
    int p = B->n;

    int len_a, len_b;

    int *Cp = (int *) malloc((m + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);

    Cp[0] = 0;

    // --------------------------------------------------------------
    //            countz nnz and fill column indices
    // --------------------------------------------------------------
    int nnz = 0;
    for (int i = 0; i < A->m; i++)
    {
        len_a = A->p[i + 1] - A->p[i];

        for (int j = 0; j < B->n; j++)
        {
            len_b = B->p[j + 1] - B->p[j];

            if (has_overlap(A->i + A->p[i], len_a, B->i + B->p[j], len_b))
            {
                iVec_append(Ci, j);
                nnz++;
            }
        }

        Cp[i + 1] = nnz;
    }

    CSR_Matrix *C = new_csr_matrix(m, p, nnz);
    memcpy(C->p, Cp, (m + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));
    free(Cp);
    iVec_free(Ci);

    return C;
}

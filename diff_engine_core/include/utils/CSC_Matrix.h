#ifndef CSC_MATRIX_H
#define CSC_MATRIX_H

#include "CSR_Matrix.h"

/* CSC (Compressed Sparse Column) Matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - p: array of size (n + 1) indicating start of each column
 * - i: array of size nnz containing row indices
 * - x: array of size nnz containing values
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct CSC_Matrix
{
    int *p;
    int *i;
    double *x;
    int m;
    int n;
    int nnz;
} CSC_Matrix;

/* Allocate a new CSC matrix with given dimensions and nnz */
CSC_Matrix *new_csc_matrix(int m, int n, int nnz);

/* Free a CSC matrix */
void free_csc_matrix(CSC_Matrix *matrix);

CSC_Matrix *csr_to_csc(const CSR_Matrix *A);

/* Allocate sparsity pattern for C = A^T D A for diagonal D */
CSR_Matrix *ATA_alloc(const CSC_Matrix *A);

/* Allocate sparsity pattern for C = B^T D A for diagonal D */
CSR_Matrix *BTA_alloc(const CSC_Matrix *A, const CSC_Matrix *B);

/* Compute values for C = A^T D A. C must have precomputed sparsity pattern  */
void ATDA_fill_values(const CSC_Matrix *A, const double *d, CSR_Matrix *C);

/* Compute values for C = B^T D A. C must have precomputed sparsity pattern  */
void BTDA_fill_values(const CSC_Matrix *A, const CSC_Matrix *B, const double *d,
                      CSR_Matrix *C);

/* C = z^T A where A is in CSC format and C is assumed to have one row.
 * C must have column indices pre-computed. Fills in values of C only.
 */
void csc_matvec_fill_values(const CSC_Matrix *A, const double *z, CSR_Matrix *C);

CSC_Matrix *csr_to_csc_fill_sparsity(const CSR_Matrix *A, int *iwork);
void csr_to_csc_fill_values(const CSR_Matrix *A, CSC_Matrix *C, int *iwork);

/* Allocate CSR matrix for C = A @ B where A is CSR, B is CSC
 * Precomputes sparsity pattern. No workspace required.
 */
CSR_Matrix *csr_csc_matmul_alloc(const CSR_Matrix *A, const CSC_Matrix *B);

/* Fill values of C = A @ B where A is CSR, B is CSC
 * C must have sparsity pattern already computed
 */
void csr_csc_matmul_fill_values(const CSR_Matrix *A, const CSC_Matrix *B,
                                CSR_Matrix *C);

#endif /* CSC_MATRIX_H */

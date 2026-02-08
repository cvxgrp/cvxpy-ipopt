#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H
#include <stdbool.h>

/* forward declaration */
struct int_double_pair;

/* CSR (Compressed Sparse Row) Matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - p: array of size (m + 1) indicating start of each row
 * - i: array of size nnz containing column indices
 * - x: array of size nnz containing values
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct CSR_Matrix
{
    int *p;
    int *i;
    double *x;
    int m;
    int n;
    int nnz;
} CSR_Matrix;

/* Allocate a new CSR matrix with given dimensions and nnz */
CSR_Matrix *new_csr_matrix(int m, int n, int nnz);

/* Free a CSR matrix */
void free_csr_matrix(CSR_Matrix *matrix);

/* Copy CSR matrix A to C */
void copy_csr_matrix(const CSR_Matrix *A, CSR_Matrix *C);

/* Build block-diagonal repeat A_blk = I_p kron A. Returns newly allocated CSR
 * matrix of size (p*A->m) x (p*A->n) with nnz = p*A->nnz. */
CSR_Matrix *block_diag_repeat_csr(const CSR_Matrix *A, int p);

/* Build left-repeated Kronecker A_kron = A kron I_p. Returns newly allocated CSR
 * matrix of size (A->m * p) x (A->n * p) with nnz = A->nnz * p. */
CSR_Matrix *kron_identity_csr(const CSR_Matrix *A, int p);

/* matvec y = Ax, where A indices minus col_offset gives x indices. Returns y as
 * dense. */
void csr_matvec(const CSR_Matrix *A, const double *x, double *y, int col_offset);
void csr_matvec_wo_offset(const CSR_Matrix *A, const double *x, double *y);

/* C = z^T A is assumed to have one row. C must have column indices pre-computed
and transposed matrix AT must be provided. Fills in values of C only.
 */
void csr_matvec_fill_values(const CSR_Matrix *AT, const double *z, CSR_Matrix *C);

/* Insert value into CSR matrix A with just one row at col_idx. Assumes that A
has enough space and that A does not have an element at col_idx. It does update
nnz. */
void csr_insert_value(CSR_Matrix *A, int col_idx, double value);

/* Compute C = diag(d) * A where d is an array and A, C are CSR matrices
 * d must have length m
 * C must be pre-allocated with same dimensions as A */
void diag_csr_mult(const double *d, const CSR_Matrix *A, CSR_Matrix *C);
void diag_csr_mult_fill_values(const double *d, const CSR_Matrix *A, CSR_Matrix *C);

/* Compute C = A + B where A, B, C are CSR matrices
 * A and B must have same dimensions
 * C must be pre-allocated with sufficient nnz capacity.
 * C must be different from A and B */
void sum_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C);
/* Compute sparsity pattern of A + B where A, B, C are CSR matrices.
 * Fills C->p, C->i, and C->nnz; does not touch C->x. */
void sum_csr_matrices_fill_sparsity(const CSR_Matrix *A, const CSR_Matrix *B,
                                    CSR_Matrix *C);

/* Fill only the values of C = A + B, assuming C's sparsity pattern (p and i)
 * is already filled and matches the union of A and B per row. Does not modify
 * C->p, C->i, or C->nnz. */
void sum_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                  CSR_Matrix *C);

/* Compute C = diag(d1) * A + diag(d2) * B where A, B, C are CSR matrices */
void sum_scaled_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C,
                             const double *d1, const double *d2);

/* Fill only the values of C = diag(d1) * A + diag(d2) * B, assuming C's sparsity
 * pattern (p and i) is already filled and matches the union of A and B per row.
 * Does not modify C->p, C->i, or C->nnz. */
void sum_scaled_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                         CSR_Matrix *C, const double *d1,
                                         const double *d2);

/* Sum all rows of A into a single row matrix C */
void sum_all_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                      struct int_double_pair *pairs);

/* iwork must have size max(C->n, A->nnz), and idx_map must have size A->nnz. */
void sum_all_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A, CSR_Matrix *C,
                                                int *iwork, int *idx_map);

/* Fill values of summed rows using precomputed idx_map and sparsity of C */
// void sum_all_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
//                                  const int *idx_map);

/* Fill accumulator for summing rows using precomputed idx_map for each nnz of A.
   Must memset accumulator to zero before calling. */
void idx_map_accumulator(const CSR_Matrix *A, const int *idx_map,
                         double *accumulator);
void idx_map_accumulator_with_spacing(const CSR_Matrix *A, const int *idx_map,
                                      double *accumulator, int spacing);

/* Sum blocks of rows of A into a matrix C */
void sum_block_of_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                           struct int_double_pair *pairs, int row_block_size);

/* Build sparsity and index map for summing blocks of rows.
 * iwork must have size max(A->n, A->nnz), and idx_map must have size A->nnz. */
void sum_block_of_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                     CSR_Matrix *C,
                                                     int row_block_size, int *iwork,
                                                     int *idx_map);

/* Fill values for summing blocks of rows using precomputed idx_map */
// void sum_block_of_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
//                                        const int *idx_map);

/* Sum evenly spaced rows of A into a matrix C */
void sum_evenly_spaced_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                struct int_double_pair *pairs, int row_spacing);

/* Build sparsity and index map for summing evenly spaced rows.
 * iwork must have size max(A->n, A->nnz), and idx_map must have size A->nnz. */
void sum_evenly_spaced_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                          CSR_Matrix *C,
                                                          int row_spacing,
                                                          int *iwork, int *idx_map);

/* Fill values for summing evenly spaced rows using precomputed idx_map */
// void sum_evenly_spaced_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
//                                             const int *idx_map);

/* Sum evenly spaced rows of A starting at offset into a row matrix C */
void sum_spaced_rows_into_row_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                  struct int_double_pair *pairs, int offset,
                                  int spacing);
/* Fills the sparsity and index map for summing spaced rows into a row matrix */
void sum_spaced_rows_into_row_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                            CSR_Matrix *C,
                                                            int spacing, int *iwork,
                                                            int *idx_map);

/* Count number of columns with nonzero entries */
int count_nonzero_cols(const CSR_Matrix *A, bool *col_nz);

/* inserts 'idx' into array 'arr' in sorted order, and moves the other elements */
void insert_idx(int idx, int *arr, int len);

double csr_get_value(const CSR_Matrix *A, int row, int col);

/* iwork must be of size A->n*/
CSR_Matrix *transpose(const CSR_Matrix *A, int *iwork);
CSR_Matrix *AT_alloc(const CSR_Matrix *A, int *iwork);

/* Fill values of A^T given sparsity pattern is already computed */
void AT_fill_values(const CSR_Matrix *A, CSR_Matrix *AT, int *iwork);

/* Expand symmetric CSR matrix A to full matrix C. A is assumed to store
   only upper triangle. C must be pre-allocated with sufficient nnz */
void symmetrize_csr(const int *Ap, const int *Ai, int m, CSR_Matrix *C);

#endif /* CSR_MATRIX_H */

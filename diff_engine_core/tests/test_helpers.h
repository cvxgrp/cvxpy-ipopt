#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "expr.h"

/* Compare two double arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_double_array(const double *actual, const double *expected, int size);

/* Compare two int arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_int_array(const int *actual, const int *expected, int size);

#endif /* TEST_HELPERS_H */

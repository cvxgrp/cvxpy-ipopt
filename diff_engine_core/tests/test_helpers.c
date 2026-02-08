#include <math.h>
#include <stdio.h>

#include "expr.h"

#define EPSILON 1e-7

#define ABS_TOL 1e-6
#define REL_TOL 1e-6

int is_equal_double(double a, double b)
{
    return fabs(a - b) <= fmax(ABS_TOL, REL_TOL * fmax(fabs(a), fabs(b)));
}

int cmp_double_array(const double *actual, const double *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (!is_equal_double(actual[i], expected[i]))
        {
            printf("  FAILED: actual[%d] = %f, expected %f\n", i, actual[i],
                   expected[i]);
            return 0;
        }
    }
    return 1;
}

int cmp_int_array(const int *actual, const int *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (actual[i] != expected[i])
        {
            printf("  FAILED: actual[%d] = %d, expected %d\n", i, actual[i],
                   expected[i]);
            return 0;
        }
    }
    return 1;
}

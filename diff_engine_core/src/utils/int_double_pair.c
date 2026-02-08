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
#include "utils/int_double_pair.h"
#include <stdlib.h>

static int compare_int_double_pair(const void *a, const void *b)
{
    const int_double_pair *pair_a = (const int_double_pair *) a;
    const int_double_pair *pair_b = (const int_double_pair *) b;

    if (pair_a->col < pair_b->col) return -1;
    if (pair_a->col > pair_b->col) return 1;
    return 0;
}

int_double_pair *new_int_double_pair_array(int size)
{
    return (int_double_pair *) malloc(size * sizeof(int_double_pair));
}

void set_int_double_pair_array(int_double_pair *pair, int *ints, double *doubles,
                               int size)
{
    for (int k = 0; k < size; k++)
    {
        pair[k].col = ints[k];
        pair[k].val = doubles[k];
    }
}

void free_int_double_pair_array(int_double_pair *array)
{
    free(array);
}

void sort_int_double_pair_array(int_double_pair *array, int size)
{
    qsort(array, size, sizeof(int_double_pair), compare_int_double_pair);
}

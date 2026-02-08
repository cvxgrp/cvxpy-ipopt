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

#ifndef INT_DOUBLE_PAIR_H
#define INT_DOUBLE_PAIR_H

typedef struct int_double_pair
{
    int col;
    double val;
} int_double_pair;

int_double_pair *new_int_double_pair_array(int size);
void set_int_double_pair_array(int_double_pair *pair, int *ints, double *doubles,
                               int size);
void free_int_double_pair_array(int_double_pair *array);
void sort_int_double_pair_array(int_double_pair *array, int size);

#endif /* INT_DOUBLE_PAIR_H */

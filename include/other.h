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
#ifndef OTHER_H
#define OTHER_H

#include "expr.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"

expr *new_quad_form(expr *child, CSR_Matrix *Q);

/* product of all entries, without axis argument */
expr *new_prod(expr *child);

/* product of entries along axis=0 (columnwise products) */
expr *new_prod_axis_zero(expr *child);

/* product of entries along axis=1 (rowwise products) */
expr *new_prod_axis_one(expr *child);

#endif /* OTHER_H */

# DNLP Diff Engine - Atom Status Tracking

This file tracks atoms implemented in the DNLP diff engine (`DNLP-diff-engine/`)
and test coverage for NLP tests.

## Implemented Atoms (Python bindings available)

### Elementwise Univariate
- [x] log
- [x] exp
- [x] entr
- [x] power
- [x] sqrt (via power with p=0.5)
- [x] logistic
- [x] xexp
- [x] sin, cos, tan
- [x] sinh, tanh, asinh, atanh

### Affine
- [x] variable
- [x] constant
- [x] add (AddExpression)
- [x] neg (NegExpression)
- [x] sum (Sum)
- [x] promote (Promote)
- [x] index (index, special_index)
- [x] reshape (Fortran order only)

### Bivariate
- [x] multiply (elementwise, with const scalar/vector variants)
- [x] quad_form (QuadForm)
- [x] quad_over_lin
- [x] rel_entr (equal-sized args only)

### Matrix Operations
- [x] left_matmul (A @ f(x) where A is constant)
- [x] right_matmul (f(x) @ A where A is constant)

## Missing Atoms (needed for full test coverage)

### High Priority
- [ ] broadcast_to - needed for test_localization, test_row_broadcast, test_circle_packing_best_of
- [ ] Prod - needed for 9 prod IPOPT tests
- [ ] MulExpression (bivariate matmul) - f(x) @ g(x) where both matrices depend on variables

### Medium Priority
- [ ] rel_entr scalar variants (first_arg_scalar, second_arg_scalar) - declared but not implemented in C
- [ ] hstack - C code exists, needs Python binding

### Low Priority / Not Needed Yet
- [ ] trace - C code exists but sparsity pattern not computed in init
- [ ] cosh, acos, asin, atan, acosh - trig/hyperbolic variants

## Test Results Summary

### test_nlp_solvers.py (15 tests)
| Test | Status | Notes |
|------|--------|-------|
| test_hs071 | PASS | |
| test_mle | PASS | |
| test_portfolio_opt | PASS | |
| test_rosenbrock | PASS | |
| test_qcp | PASS | |
| test_analytic_polytope_center | PASS | |
| test_socp | PASS | |
| test_portfolio_socp | PASS | |
| test_geo_mean | PASS | |
| test_geo_mean2 | PASS | |
| test_clnlbeam | PASS | |
| test_circle_packing_formulation_one | PASS | |
| test_circle_packing_formulation_two | PASS | |
| test_circle_packing_formulation_three | PASS | |
| test_localization | FAIL | needs broadcast_to |

### test_scalar_and_matrix_problems.py (24 tests)
- **23 passing**
- **1 failing**: test_rel_entr_matrix_variable_and_scalar_variable (needs rel_entr scalar variants)

### test_entropy_related.py (8 tests)
- **6 passing**
- **2 failing**: test_KL_three_graph_form, test_KL_three_not_graph_form (need bivariate matmul)

### test_matmul.py (6 tests)
- **2 passing**: test_matmul_with_function_right, test_matmul_with_function_left
- **3 failing**: need MulExpression with two non-constant args
- **1 XFAIL**: test_matmul_same_variable (expected)

### test_prod.py (14 tests)
- **5 passing**: DNLP rule tests
- **9 failing**: IPOPT tests need Prod atom

### test_broadcast.py (3 tests)
- **1 passing**: test_scalar_to_matrix
- **2 failing**: test_row_broadcast, test_column_broadcast (need broadcast_to)

### test_best_of.py (2 tests)
- **2 failing**: test_circle_packing_best_of_one, test_path_planning_best_of_two (need broadcast_to)

### Fully Passing Test Files
- test_dnlp.py (10 tests)
- test_log_sum_exp.py (4 tests)
- test_risk_parity.py (4 tests)
- test_abs.py (4 tests)
- test_Sharpe_ratio.py (1 test)
- test_problem.py (8 tests)

### Skipped Test Files
- test_interfaces.py (17 tests) - requires Knitro/COPT licenses
- test_ML_Gaussian_stress.py (2 tests) - requires stress testing setup

### Timeout/Hung Tests
- test_hyperbolic.py - hangs indefinitely
- test_huber_sum_largest.py - hangs indefinitely

## Known Issues

1. **Bivariate matmul not supported**: `f(x) @ g(x)` where both matrices depend on optimization variables is not implemented (e.g., `X @ Y` with X, Y both Variables, or `cos(X) @ sin(Y)`). Currently only `A @ f(x)` (left_matmul) and `f(x) @ A` (right_matmul) with constant A are supported.

2. **rel_entr scalar broadcasting**: The C implementation only handles equal-sized arguments. Scalar broadcasting variants are declared in header but not implemented.

3. **Missing Python bindings**: hstack needs `make_*` function added to `python/bindings.c`.

## Recent Changes

### 2025-01-14 (diffengine-setup branch)
- `81c5a12` Reshape atom fixed - adds make_reshape Python binding
- All circle packing tests now PASS
- test_clnlbeam now PASSES (previously segfaulted)
- Memory issues with many constraints fixed
- **14 of 15 test_nlp_solvers.py tests now passing**

### 2025-01-12 (indexing branch)
- `a8fa3dd` Add rel_entr binding and converter
- `0b7e41a` Add quad_over_lin binding and converter
- `c69620a` Add reshape converter (Fortran order only)
- `6af82bf` Add sqrt converter as power with p=0.5
- `5559dfd` Add index atom for array indexing and slicing

## Integration Status
- nlp_solver.py Oracles class uses C_problem wrapper from diff engine
- Sparse matrix format conversion (CSR to COO) working
- Sparsity pattern caching working

### Full NLP Test Suite Summary (2025-01-14)
- **259 passed, 20 failed, 64 skipped, 1 xfailed** (all nlp_tests)
- **14 of 15 test_nlp_solvers.py tests passing** (IPOPT)
- Remaining failures need: broadcast_to (5), Prod (9), MulExpression (5), rel_entr scalar (1)

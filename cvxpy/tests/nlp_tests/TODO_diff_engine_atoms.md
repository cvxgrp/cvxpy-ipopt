# TODO: Atoms to Implement in Diff Engine

This file tracks atoms that need to be implemented in the DNLP diff engine
(`DNLP-diff-engine/`) to support all NLP tests.

## Currently Supported Atoms

### Affine
- `NegExpression` - negation
- `Promote` - promote scalar to vector/matrix
- `AddExpression` - addition (n-ary)
- `Sum` - summation

### Elementwise Univariate
- `log` - logarithm
- `exp` - exponential
- `power` - power function (x^p)
- `sin`, `cos`, `tan` - trigonometric
- `sinh`, `tanh`, `asinh`, `atanh` - hyperbolic
- `entr` - entropy (-x*log(x))
- `logistic` - logistic function
- `xexp` - x*exp(x)

### Bivariate
- `multiply` - elementwise multiplication
- `MulExpression` - matrix multiplication (A @ x or x @ A)

## Missing Atoms (Blocking Tests)

### Priority 1: Core Operations

- [ ] `DivExpression` - division (1/x or const/x)
  - Used in: test_mle
  - Workaround: Could use `multiply` with `power(x, -1)`

- [ ] `index` - array indexing (x[i])
  - Used in: test_rosenbrock, test_hs071, test_socp, test_circle_packing, test_clnlbeam
  - Complex to implement - requires slicing support
  - Blocks many tests

### Priority 2: Quadratic Forms

- [ ] `QuadForm` - quadratic form (x^T P x)
  - Used in: test_portfolio_opt
  - Could potentially be implemented as composition of matmul + sum

- [ ] `quad_over_lin` - quadratic over linear
  - C implementation exists: `new_quad_over_lin`
  - Python bindings needed

### Priority 3: Norms and Special Functions

- [ ] `norm` (L2) - Euclidean norm
- [ ] `norm1` - L1 norm
- [ ] `norm_inf` - infinity norm
- [ ] `sum_squares` - sum of squares
- [ ] `sqrt` - square root (can use power(x, 0.5))

### Priority 4: Other

- [ ] `geo_mean` - geometric mean (causes segfault - C bug)
- [ ] `rel_entr` - relative entropy
  - C implementation exists: `new_rel_entr_*`
  - Python bindings needed
- [ ] `huber` - Huber loss
- [ ] `abs` - absolute value (ESR)
- [ ] `max` - maximum (ESR)
- [ ] `min` - minimum (HSR)

## Test Status Summary

| Test | Status | Blocking Issue |
|------|--------|----------------|
| test_qcp | PASS | - |
| test_analytic_polytope_center | PASS | - |
| test_hs071 | FAIL | index (x[0], x[1], etc.) |
| test_mle | SEGFAULT | DivExpression or C bug |
| test_portfolio_opt | FAIL | QuadForm |
| test_rosenbrock | FAIL | index (x[0], x[1]) |
| test_socp | FAIL | index, norm |
| test_portfolio_socp | FAIL | norm |
| test_localization | FAIL | Unknown |
| test_circle_packing_* | FAIL | index, norm |
| test_geo_mean | SEGFAULT | C bug in init_derivatives |
| test_geo_mean2 | SEGFAULT | C bug in init_derivatives |
| test_clnlbeam | FAIL | index |

## Summary

- **2/15 IPOPT tests now pass** (test_qcp, test_analytic_polytope_center)
- All tests skip for KNITRO/UNO/COPT (not installed)
- Several tests cause segfaults - likely C library bugs that need investigation
- Main blockers are:
  1. **index** - array indexing, blocks ~7 tests
  2. **norm** - Euclidean norm, blocks ~4 tests
  3. **C bugs** - segfaults in some expression trees

## Recently Added (2025-01-11)

### Python Bindings Added
- `power` - now working, test_qcp passes
- `multiply` - working
- `sin`, `cos`, `tan` - working
- `sinh`, `tanh`, `asinh`, `atanh` - bindings added
- `entr`, `logistic`, `xexp` - bindings added
- `left_matmul`, `right_matmul` - added, test_analytic_polytope_center passes

### Integration Status
- nlp_solver.py Oracles class fully replaced with C_problem wrapper
- Sparse matrix format conversion (CSR to COO) working
- Sparsity pattern caching working

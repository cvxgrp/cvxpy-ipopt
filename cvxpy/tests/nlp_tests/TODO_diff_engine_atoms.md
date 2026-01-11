# TODO: Atoms to Implement in Diff Engine

This file tracks atoms that need to be implemented in the DNLP diff engine
(`DNLP-diff-engine/`) to support all NLP tests.

## Currently Supported Atoms

- `log` - logarithm
- `exp` - exponential
- `NegExpression` - negation
- `Promote` - promote scalar to vector/matrix
- `AddExpression` - addition (n-ary)
- `Sum` - summation

## Missing Atoms (Blocking Tests)

### Priority 1: Core Operations (blocks most tests)

- [ ] `multiply` - elementwise multiplication
  - Used in: test_hs071, test_mle, test_clnlbeam, and many others
  - C implementation needed: `make_multiply(left, right)`

- [ ] `power` - power function (x^p)
  - Used in: test_rosenbrock, test_qcp, test_clnlbeam
  - C implementation needed: `make_power(base, exponent)`

- [ ] `MulExpression` - matrix multiplication (@)
  - Used in: test_analytic_polytope_center, test_portfolio_socp
  - C implementation needed: `make_matmul(left, right)`

- [ ] `index` - array indexing (x[i])
  - Used in: test_analytic_polytope_center, test_circle_packing
  - C implementation needed: `make_index(expr, indices)`

### Priority 2: Quadratic Forms

- [ ] `QuadForm` - quadratic form (x^T P x)
  - Used in: test_portfolio_opt
  - Could potentially be implemented as composition of matmul + sum

### Priority 3: Norms and Special Functions

- [ ] `norm` (L2) - Euclidean norm
- [ ] `norm1` - L1 norm
- [ ] `norm_inf` - infinity norm
- [ ] `sum_squares` - sum of squares
- [ ] `sqrt` - square root

### Priority 4: Trigonometric

- [ ] `sin` - sine
- [ ] `cos` - cosine
- [ ] `tan` - tangent

### Priority 5: Other

- [ ] `geo_mean` - geometric mean
- [ ] `entr` - entropy (-x*log(x))
- [ ] `rel_entr` - relative entropy
- [ ] `logistic` - logistic function
- [ ] `huber` - Huber loss
- [ ] `abs` - absolute value (ESR)
- [ ] `max` - maximum (ESR)
- [ ] `min` - minimum (HSR)

## Test Status Summary

| Test | Missing Atom(s) | Status |
|------|-----------------|--------|
| test_hs071 | multiply | BLOCKED |
| test_mle | multiply | BLOCKED |
| test_portfolio_opt | QuadForm | BLOCKED |
| test_rosenbrock | power | BLOCKED |
| test_qcp | power | BLOCKED |
| test_analytic_polytope_center | MulExpression, index | BLOCKED |
| test_socp | MulExpression, index | BLOCKED |
| test_portfolio_socp | MulExpression | BLOCKED |
| test_localization | power, multiply | BLOCKED |
| test_circle_packing_* | index, power, multiply | BLOCKED |
| test_geo_mean | power, multiply | BLOCKED |
| test_geo_mean2 | power, multiply | BLOCKED |
| test_clnlbeam | power, multiply | BLOCKED |

## Notes

- All tests skip for KNITRO/UNO/COPT (not installed)
- 15/15 IPOPT tests currently fail due to missing atoms
- Integration with nlp_solver.py is complete and working
- Simple problems using only log/exp/sum work correctly

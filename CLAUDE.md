# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CVXPY is a Python-embedded modeling language for convex optimization. This fork extends CVXPY with **Disciplined Nonlinear Programming (DNLP)** support, enabling general smooth nonlinear optimization problems to be solved via NLP solvers like IPOPT, Knitro, COPT, and Uno.

For the theoretical foundation, see the paper: [Disciplined Nonlinear Programming](https://web.stanford.edu/~boyd/papers/dnlp.html).

## Build and Development Commands

```bash
# Install IPOPT solver (required for NLP - use conda, NOT pip)
conda install -c conda-forge cyipopt

# Install from source (development mode)
pip install -e .

# Run all tests
pytest cvxpy/tests/

# Run a specific test file
pytest cvxpy/tests/test_dgp.py

# Run a specific test method
pytest cvxpy/tests/test_dgp.py::TestDgp::test_product

# Run NLP-specific tests
pytest cvxpy/tests/nlp_tests/

# Lint with ruff
ruff check cvxpy

# Auto-fix lint issues
ruff check --fix cvxpy

# Build documentation
cd doc && make html

# Run tests with stdout visible (useful for debugging)
pytest -s cvxpy/tests/test_file.py

# Run tests with verbose output
pytest -v cvxpy/tests/test_file.py

# Run tests that match a pattern
pytest -k "test_ipopt" cvxpy/tests/nlp_tests/

# Run tests and stop on first failure
pytest -x cvxpy/tests/
```

## Supported NLP Solvers

| Solver | License | Installation |
|--------|---------|--------------|
| [IPOPT](https://github.com/coin-or/Ipopt) | EPL-2.0 | `conda install -c conda-forge cyipopt` |
| [Knitro](https://www.artelys.com/solvers/knitro/) | Commercial | `pip install knitro` (requires license) |
| [COPT](https://www.copt.de/) | Commercial | Requires license |
| [Uno](https://github.com/cuter-testing/uno) | Open source | See Uno documentation |

## DNLP Diff Engine (Subproject)

The `DNLP-diff-engine/` directory contains a separate C library with Python bindings that provides automatic differentiation for NLP problems. It builds expression trees from CVXPY problems and computes derivatives (gradients, Jacobians, Hessians) for NLP solvers. See `DNLP-diff-engine/CLAUDE.md` for details on that subproject.

## Solving with DNLP

To solve a problem as a DNLP (rather than DCP):
```python
import cvxpy as cp
import numpy as np

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(objective), constraints)

# Initial point required for NLP solvers
x.value = np.ones(n)

# Solve with nlp=True
prob.solve(nlp=True, solver=cp.IPOPT)

# Optional: Run multiple solves with random initial points, return best
prob.solve(nlp=True, solver=cp.IPOPT, best_of=5)
```

### Solver Options

```python
# IPOPT options
prob.solve(nlp=True, solver=cp.IPOPT, max_iter=1000, tol=1e-8)

# Knitro algorithm variants
prob.solve(nlp=True, solver="knitro_ipm")   # Interior point method
prob.solve(nlp=True, solver="knitro_sqp")   # Sequential quadratic programming
prob.solve(nlp=True, solver="knitro_alm")   # Augmented Lagrangian method

# Uno presets
prob.solve(nlp=True, solver="uno_ipm")   # IPOPT-like IPM
prob.solve(nlp=True, solver="uno_sqp")   # Filter SQP method
```

## Architecture

### Expression System

Expressions form an AST (Abstract Syntax Tree):
- **Expression** (base) → Variable, Parameter, Constant, Atom
- **Atom** subclasses implement mathematical functions (in `cvxpy/atoms/`)
- Each atom defines curvature, sign, and disciplined programming rules

### Problem Types

CVXPY supports multiple disciplined programming paradigms:
- **DCP** (Disciplined Convex Programming) - standard convex problems
- **DGP** (Disciplined Geometric Programming) - geometric programs
- **DQCP** (Disciplined Quasiconvex Programming) - quasiconvex programs
- **DNLP** (Disciplined Nonlinear Programming) - smooth nonlinear programs (this fork's extension)

### Reduction Pipeline

Problems are transformed through a chain of reductions before solving:
```
Problem → [Reductions] → Canonical Form → Solver
```

Key reduction classes in `cvxpy/reductions/`:
- `Reduction` base class with `accepts()`, `apply()`, `invert()` methods
- `Chain` composes multiple reductions
- `SolvingChain` orchestrates the full solve process

For DNLP: `CvxAttr2Constr` → `Dnlp2Smooth` → `NLPSolver`

### Solver Categories

- **ConicSolvers** (`cvxpy/reductions/solvers/conic_solvers/`) - SCS, Clarabel, ECOS, etc.
- **QPSolvers** (`cvxpy/reductions/solvers/qp_solvers/`) - OSQP, ProxQP, etc.
- **NLPSolvers** (`cvxpy/reductions/solvers/nlp_solvers/`) - IPOPT, Knitro, COPT, Uno

### NLP System (DNLP Extension)

The NLP infrastructure provides oracle-based interfaces for nonlinear solvers:
- `nlp_solver.py` - Base `NLPsolver` class with:
  - `Bounds` class: extracts variable bounds (`lb`, `ub`) and constraint bounds (`cl`, `cu`) from problem. Transforms the problem to canonical form (Equality → zero constraints, Inequality/NonPos → nonneg form), creating a `new_problem` attribute used by the solver
  - `Oracles` class: provides function and derivative oracles for NLP solvers:
    - `objective(x)`, `gradient(x)` - objective function and its gradient
    - `constraints(x)`, `jacobian(x)` - constraint functions and Jacobian
    - `jacobianstructure()` - sparsity pattern of Jacobian (row, col indices)
    - `hessian(x, obj_factor, lambda)` - Lagrangian Hessian
    - `hessianstructure()` - sparsity pattern of Hessian (lower triangular)
- `dnlp2smooth.py` - Transforms DNLP problems to smooth form via `Dnlp2Smooth` reduction
- DNLP validation: expressions must be smooth (ESR and HSR - essentially smooth and respecting)
- Problem validity checked via `problem.is_dnlp()` method

**Oracle implementations:** The `Oracles` class uses CVXPY expression methods (`expr.jacobian()`, `expr.hess_vec()`) to compute derivatives. The `DNLP-diff-engine/` subproject provides an alternative C-based implementation for performance-critical applications.

### Adding DNLP Support for New Atoms

To add DNLP support for a new atom:
1. Create a canonicalizer in `cvxpy/reductions/dnlp2smooth/canonicalizers/` (e.g., `myatom_canon.py`)
2. The canonicalizer converts non-smooth atoms to smooth equivalents using auxiliary variables
3. Register the canonicalizer in `canonicalizers/__init__.py` by adding to the `SMOOTH_CANON_METHODS` dict (maps atom class → canonicalizer function)
4. Ensure the atom has proper `is_smooth()`, `is_esr()`, `is_hsr()` methods defined in the atom class

### DNLP Rules (ESR/HSR)

DNLP extends DCP by allowing smooth and piecewise-smooth functions:
- **Smooth**: functions that are both ESR and HSR (analogous to affine in DCP)
- **ESR** (Essentially Smooth Respecting): can be minimized or appear in `<= 0` constraints (e.g., `abs`, `max`, `norm1`)
- **HSR** (Hierarchically Smooth Respecting): can be maximized or appear in `>= 0` constraints (e.g., `min`, `sqrt`)

Use `expr.is_smooth()`, `expr.is_esr()`, `expr.is_hsr()` to check expression properties.

NLP tests are in `cvxpy/tests/nlp_tests/` with Jacobian and Hessian verification tests.

For more examples, see [DNLP-examples](https://github.com/cvxgrp/dnlp-examples).

## Key Directories

- `cvxpy/atoms/` - Mathematical functions (elementwise, affine, etc.)
- `cvxpy/expressions/` - Expression base classes and variable types
- `cvxpy/problems/` - Problem class and solving logic
- `cvxpy/reductions/` - Problem transformation pipeline
- `cvxpy/reductions/solvers/nlp_solvers/` - NLP solver interfaces (ipopt_nlpif.py, knitro_nlpif.py, etc.)
- `cvxpy/reductions/dnlp2smooth/canonicalizers/` - Smooth canonicalizers for individual atoms
- `cvxpy/tests/nlp_tests/` - NLP-specific tests

## Code Style

- Uses ruff for linting (configured in `pyproject.toml`)
- Target Python version: 3.11+
- Line length: 100 characters
- Pre-commit hooks available: `pip install pre-commit && pre-commit install`

## License Header

New files should include the Apache 2.0 license header:
```python
"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
```

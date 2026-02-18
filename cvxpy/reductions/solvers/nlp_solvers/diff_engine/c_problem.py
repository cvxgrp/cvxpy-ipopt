"""Wrapper around C problem struct for CVXPY problems.

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

import numpy as np
from scipy import sparse

import cvxpy as cp

# Import the low-level C bindings
try:
    from sparsediffpy import _sparsediffengine as _diffengine
except ImportError as e:
    raise ImportError(
        "NLP support requires sparsediffpy. Install with: pip install sparsediffpy"
    ) from e

from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
    convert_expressions,
)


class C_problem:
    """Wrapper around C problem struct for CVXPY problems."""

    def __init__(self, cvxpy_problem: cp.Problem, verbose: bool = True):
        c_obj, c_constraints, param_capsules, all_params = (
            convert_expressions(cvxpy_problem)
        )
        self._capsule = _diffengine.make_problem(c_obj, c_constraints, verbose)

        # Register parameters with the C problem
        if param_capsules:
            _diffengine.problem_register_params(
                self._capsule, param_capsules
            )

        self._all_params = all_params
        self._jacobian_allocated = False
        self._hessian_allocated = False

        # Push initial parameter values to C
        self.update_params()

    def init_jacobian(self):
        """Initialize Jacobian structures only. Must be called before jacobian()."""
        _diffengine.problem_init_jacobian(self._capsule)
        self._jacobian_allocated = True

    def init_hessian(self):
        """Initialize Hessian structures only. Must be called before hessian()."""
        _diffengine.problem_init_hessian(self._capsule)
        self._hessian_allocated = True

    def objective_forward(self, u: np.ndarray) -> float:
        """Evaluate objective. Returns obj_value float."""
        return _diffengine.problem_objective_forward(self._capsule, u)

    def constraint_forward(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraints only. Returns constraint_values array."""
        return _diffengine.problem_constraint_forward(self._capsule, u)

    def gradient(self) -> np.ndarray:
        """Compute gradient of objective. Call objective_forward first. Returns gradient array."""
        return _diffengine.problem_gradient(self._capsule)

    def jacobian(self) -> sparse.csr_matrix:
        """Compute constraint Jacobian. Call constraint_forward first."""
        data, indices, indptr, shape = _diffengine.problem_jacobian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def get_jacobian(self) -> sparse.csr_matrix:
        """Get constraint Jacobian. This function does not evaluate the jacobian. """
        data, indices, indptr, shape = _diffengine.get_jacobian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def hessian(self, obj_factor: float, lagrange: np.ndarray) -> sparse.csr_matrix:
        """Compute Lagrangian Hessian.

        Computes: obj_factor * H_obj + sum(lagrange_i * H_constraint_i)

        Call objective_forward and constraint_forward before this.

        Args:
            obj_factor: Weight for objective Hessian
            lagrange: Array of Lagrange multipliers (length = total_constraint_size)

        Returns:
            scipy CSR matrix of shape (n_vars, n_vars)
        """
        data, indices, indptr, shape = _diffengine.problem_hessian(
            self._capsule, obj_factor, lagrange
        )
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def get_hessian(self) -> sparse.csr_matrix:
        """Get Lagrangian Hessian. This function does not evaluate the hessian."""
        data, indices, indptr, shape = _diffengine.get_hessian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def update_params(self):
        """Read current .value from each Parameter and push to C.

        Call this after changing parameter values to update the C expression tree
        without rebuilding it. After calling this, objective_forward/gradient/etc.
        will use the new parameter values.
        """
        if not self._all_params:
            return
        theta = np.empty(sum(p.size for p in self._all_params))
        offset = 0
        for param in self._all_params:
            original = param.leaf_of_provenance()
            if original is not None and original.sparse_idx is not None:
                # Sparse param used in fused matmul: values must be in CSR
                # data order so that refresh_param_values memcpys directly
                # into the CSR .x array.
                rows, cols = original.sparse_idx
                coo = sparse.coo_array(
                    (param.value, (rows, cols)), shape=original.shape)
                val = coo.tocsr().data.astype(np.float64)
            else:
                val = np.asarray(param.value, dtype=np.float64).flatten(order='C')
            theta[offset:offset + param.size] = val
            offset += param.size
        _diffengine.problem_update_params(self._capsule, theta)

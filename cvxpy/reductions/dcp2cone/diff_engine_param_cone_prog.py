"""DiffEngineParamConeProg: parameter application via the C diff engine.

Instead of building sparse tensors and multiplying by the parameter vector,
this class builds a C expression tree once and evaluates derivatives directly.

Key insight (evaluate-at-zero trick): After DCP2Cone, all constraint
expressions are affine in x and the objective is linear or quadratic. So:
- Jacobian(constraints) = A (constant, since affine)
- constraint_forward(0) = b
- gradient(objective) at x=0 = q
- objective_forward(0) = d
- Hessian(objective) = P (for QP; constant since quadratic)

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
from __future__ import annotations

import numpy as np
from scipy import sparse

try:
    from sparsediffpy import _sparsediffengine as _diffengine
except ImportError as e:
    raise ImportError(
        "Diff engine backend requires sparsediffpy. Install with: pip install sparsediffpy"
    ) from e

from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
    build_parameter_dict,
    build_variable_dict,
    convert_expr,
)
from cvxpy.reductions.utilities import group_constraints


class DiffEngineParamConeProg(ParamProb):
    """Parameterized cone program backed by the C diff engine.

    Duck-types ParamConeProg for the interface consumed by
    ConicSolver._prepare_data_and_inv_data() and ConicSolver.apply().
    """

    def __init__(
        self,
        x,
        variables,
        var_id_to_col,
        ordered_cons,
        parameters,
        param_id_to_col,
        has_quad_obj,
        objective_expr,
        lower_bounds=None,
        upper_bounds=None,
        lb_tensor=None,
        ub_tensor=None,
    ):
        # Variable info.
        self.x = x
        self.variables = variables
        self.var_id_to_col = var_id_to_col
        self.id_to_var = {v.id: v for v in self.variables}

        # Constraint info.
        self.constraints = ordered_cons
        self.constr_size = sum(c.size for c in ordered_cons)
        self.constr_map = group_constraints(ordered_cons)
        self.cone_dims = ConeDims(self.constr_map)

        # Parameter info.
        self.parameters = parameters
        self.param_id_to_col = param_id_to_col
        self.id_to_param = {p.id: p for p in self.parameters}
        self.param_id_to_size = {p.id: p.size for p in self.parameters}
        self.total_param_size = sum(p.size for p in self.parameters)

        # Bounds.
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.lb_tensor = lb_tensor
        self.ub_tensor = ub_tensor

        # For compatibility: P is set to a truthy sentinel when quad objective
        # is present, since ConicSolver.apply checks `problem.P is None`.
        self.P = True if has_quad_obj else None

        # Not yet formatted (restructured) for a specific solver.
        self.formatted = False

        # Restructuring matrix set by format_constraints.
        self._restruct_mat = None

        # ---- Build C expression tree ----
        self._has_quad_obj = has_quad_obj

        var_dict, n_vars = build_variable_dict(self.variables)

        all_params = list({p.id: p for p in self.parameters}.values())
        if all_params:
            param_dict, param_capsules = build_parameter_dict(
                all_params, n_vars
            )
        else:
            param_dict, param_capsules = None, []

        self._all_params = all_params

        # Convert objective.
        c_objective = convert_expr(objective_expr, var_dict, n_vars, param_dict)

        # Convert constraints: decompose into individual arg expressions.
        c_constraints = []
        for constr in ordered_cons:
            for arg in constr.args:
                c_expr = convert_expr(arg, var_dict, n_vars, param_dict)
                c_constraints.append(c_expr)

        # Create C problem.
        self._capsule = _diffengine.make_problem(c_objective, c_constraints, False)

        # Register parameters.
        if param_capsules:
            _diffengine.problem_register_params(
                self._capsule, param_capsules
            )

        # Initialize derivative structures.
        _diffengine.problem_init_jacobian(self._capsule)
        if has_quad_obj:
            _diffengine.problem_init_hessian(self._capsule)

        # Push initial parameter values.
        self._update_params()

        # Precompute zero vector for evaluate-at-zero.
        self._zeros = np.zeros(n_vars)
        self._n_vars = n_vars

        # Compute total raw constraint size (sum of all arg sizes).
        self._raw_constr_size = sum(arg.size for c in ordered_cons for arg in c.args)

    def is_mixed_integer(self):
        return self.x.attributes['boolean'] or self.x.attributes['integer']

    def _update_params(self):
        """Push current parameter values to the C expression tree."""
        if not self._all_params:
            return
        theta = np.empty(sum(p.size for p in self._all_params))
        offset = 0
        for param in self._all_params:
            val = np.asarray(param.value, dtype=np.float64).flatten(order='F')
            theta[offset:offset + param.size] = val
            offset += param.size
        _diffengine.problem_update_params(self._capsule, theta)

    def apply_parameters(self, id_to_param_value=None, zero_offset=False,
                         keep_zeros=False, quad_obj=False):
        """Evaluate A, b, q, d (and optionally P) via the diff engine.

        Uses the evaluate-at-zero trick: since all expressions are affine in x
        after DCP canonicalization, evaluating at x=0 gives the constant
        offsets, and the Jacobian/gradient gives the linear coefficients.
        """
        # Update parameter values if provided.
        if id_to_param_value is not None:
            for param in self._all_params:
                if param.id in id_to_param_value:
                    param.value = id_to_param_value[param.id]
        self._update_params()

        # Evaluate objective at zero.
        d = _diffengine.problem_objective_forward(self._capsule, self._zeros)
        q = _diffengine.problem_gradient(self._capsule)

        # Evaluate constraints at zero.
        b_raw = _diffengine.problem_constraint_forward(self._capsule, self._zeros)
        # Get Jacobian (CSR).
        jac_data, jac_indices, jac_indptr, jac_shape = _diffengine.problem_jacobian(
            self._capsule
        )
        A_raw = sparse.csr_matrix((jac_data, jac_indices, jac_indptr), shape=jac_shape)

        # Apply restructuring matrix if format_constraints was called.
        if self._restruct_mat is not None:
            # Restructuring matrix operates on the "stacked args" layout.
            A = self._restruct_mat(A_raw)
            b = np.asarray(self._restruct_mat(
                sparse.csr_matrix(b_raw.reshape(-1, 1))
            ).todense()).flatten()
        else:
            A = A_raw
            b = b_raw

        # Convert A to CSC format (expected by downstream solvers).
        if sparse.issparse(A):
            A = sparse.csc_array(A)

        # Apply parametric bounds tensors if present.
        if self.lb_tensor is not None:
            param_vec = self._build_param_vec(zero_offset)
            if param_vec is None:
                self.lower_bounds = self.lb_tensor.toarray().flatten()
            else:
                self.lower_bounds = np.asarray(
                    self.lb_tensor @ param_vec).flatten()
        if self.ub_tensor is not None:
            param_vec = self._build_param_vec(zero_offset)
            if param_vec is None:
                self.upper_bounds = self.ub_tensor.toarray().flatten()
            else:
                self.upper_bounds = np.asarray(
                    self.ub_tensor @ param_vec).flatten()

        if quad_obj:
            # Hessian of the Lagrangian with unit objective weight and zero duals.
            n_raw_constrs = self._raw_constr_size
            zero_duals = np.zeros(n_raw_constrs)
            hess_data, hess_indices, hess_indptr, hess_shape = (
                _diffengine.problem_hessian(self._capsule, 1.0, zero_duals)
            )
            P_mat = sparse.csr_matrix(
                (hess_data, hess_indices, hess_indptr), shape=hess_shape
            )
            return P_mat, q, d, A, np.atleast_1d(b)
        else:
            return q, d, A, np.atleast_1d(b)

    def _build_param_vec(self, zero_offset=False):
        """Build the parameter vector for bounds tensor application."""
        if self.total_param_size == 0:
            return None
        from cvxpy.cvxcore.python import canonInterface
        param_vec = canonInterface.get_parameter_vector(
            self.total_param_size,
            self.param_id_to_col,
            self.param_id_to_size,
            lambda idx: np.array(self.id_to_param[idx].value),
            zero_offset=zero_offset,
        )
        return param_vec


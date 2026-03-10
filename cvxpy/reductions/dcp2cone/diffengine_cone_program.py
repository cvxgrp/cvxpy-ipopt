"""
Copyright, the CVXPY authors

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

import time

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s_settings
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims, ParamConeProg
from cvxpy.reductions.matrix_stuffing import extract_mip_idx
from cvxpy.reductions.utilities import group_constraints

# Lazy import to avoid hard dependency on sparsediffpy
_diffengine = None


def _get_diffengine():
    global _diffengine
    if _diffengine is None:
        from sparsediffpy import _sparsediffengine as mod
        _diffengine = mod
    return _diffengine


class DiffengineConeProgram(ParamConeProg):
    """A cone program with concrete (non-parametric) matrices from the diff engine.

    Duck-type compatible with ParamConeProg. Stores concrete A, b, q, d, P
    matrices instead of parametric tensors.

    minimize   q'x + d + [(1/2)x'Px]
    subject to cone_constr(A*x + b) in cones
    """

    def __init__(
        self,
        x: Variable,
        A: sp.spmatrix,
        b: np.ndarray,
        q: np.ndarray,
        d: float,
        P,
        constraints: list,
        variables: list,
        var_id_to_col: dict,
        formatted: bool = False,
        lower_bounds=None,
        upper_bounds=None,
    ) -> None:
        self._A = A
        self._b = b
        self._q = q
        self._d = d

        self.x = x
        self.P = P
        self.constraints = constraints
        self.constr_size = sum(c.size for c in constraints)
        self.constr_map = group_constraints(constraints)
        self.cone_dims = ConeDims(self.constr_map)

        self.variables = variables
        self.var_id_to_col = var_id_to_col
        self.id_to_var = {v.id: v for v in self.variables}

        self.formatted = formatted
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Not parametric — empty parameter info.
        self.parameters = []
        self.param_id_to_col = {}
        self.id_to_param = {}
        self.param_id_to_size = {}
        self.total_param_size = 0

        # No parametric bound tensors.
        self.lb_tensor = None
        self.ub_tensor = None

    def is_mixed_integer(self) -> bool:
        return self.x.attributes['boolean'] or self.x.attributes['integer']

    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False, quad_obj: bool = False):
        """Return concrete matrices directly (no parameter application needed).

        A is returned in its native CSR format. Downstream consumers handle
        format conversion: QP solvers do vstack().tocsc(), and the conic path
        converts to CSC in ConicSolver.apply().
        """
        A = self._A
        if quad_obj and self.P is not None:
            return self.P, self._q, self._d, A, self._b
        return self._q, self._d, A, self._b

    def apply_restruct_mat(self, restruct_mat, restruct_mat_op=None):
        """Apply restructuring matrix to concrete A, b matrices.

        Parameters
        ----------
        restruct_mat : list
            List of sparse matrices or linear operators forming a block diagonal.
        restruct_mat_op : LinearOperator or None
            Unused for DiffengineConeProgram (uses restruct_mat directly).

        Returns
        -------
        DiffengineConeProgram
            New program with restructured A and b.
        """
        if restruct_mat:
            t0 = time.perf_counter()
            sparse_mats = []
            for mat in restruct_mat:
                if sp.issparse(mat):
                    sparse_mats.append(sp.csc_matrix(mat))
                elif callable(mat):
                    # LinearOperator or similar — materialize by applying to identity
                    eye = sp.eye_array(mat.shape[1], format='csc')
                    sparse_mats.append(sp.csc_matrix(mat(eye)))
                else:
                    eye = sp.eye_array(mat.shape[1], format='csc')
                    sparse_mats.append(sp.csc_matrix(mat @ eye))
            t1 = time.perf_counter()
            s_settings.LOGGER.info('  [apply_restruct_mat] materialize operators: %.4f s',
                                   t1 - t0)

            R = sp.block_diag(sparse_mats, format='csc')
            t2 = time.perf_counter()
            s_settings.LOGGER.info('  [apply_restruct_mat] block_diag: %.4f s', t2 - t1)

            new_A = R @ self._A
            t3 = time.perf_counter()
            s_settings.LOGGER.info('  [apply_restruct_mat] R @ A: %.4f s', t3 - t2)

            new_b = np.asarray(R @ self._b).flatten()
            t4 = time.perf_counter()
            s_settings.LOGGER.info('  [apply_restruct_mat] R @ b: %.4f s', t4 - t3)
        else:
            new_A, new_b = self._A, self._b
        return DiffengineConeProgram(
            self.x, new_A, new_b, self._q, self._d, self.P,
            self.constraints, self.variables, self.var_id_to_col,
            formatted=True,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    def split_solution(self, sltn, active_vars=None):
        from cvxpy.reductions import cvx_attr2constr
        if active_vars is None:
            active_vars = [v.id for v in self.variables]
        sltn_dict = {}
        for var_id, col in self.var_id_to_col.items():
            if var_id in active_vars:
                var = self.id_to_var[var_id]
                value = sltn[col:var.size + col]
                if var.attributes_were_lowered():
                    orig_var = var.leaf_of_provenance()
                    value = cvx_attr2constr.recover_value_for_leaf(
                        orig_var, value, project=False)
                    sltn_dict[orig_var.id] = np.reshape(
                        value, orig_var.shape, order='F')
                else:
                    sltn_dict[var_id] = np.reshape(
                        value, var.shape, order='F')
        return sltn_dict


def build_diffengine_cone_program(problem, ordered_cons, inverse_data, quad_obj):
    """Build a DiffengineConeProgram from a canonicalized problem.

    Uses the sparsediffpy diff engine to extract A, b, q, d, P by evaluating
    constraint/objective expressions at x=0 and differentiating.

    Parameters
    ----------
    problem : Problem
        The CVXPY problem (post Dcp2Cone, with affine constraints and
        linear/quadratic objective).
    ordered_cons : list
        Ordered constraints (Zero, NonNeg, SOC, PSD, ExpCone, ...).
    inverse_data : InverseData
        Inverse data for the problem.
    quad_obj : bool
        Whether the objective is quadratic.

    Returns
    -------
    DiffengineConeProgram
    """
    from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
        build_variable_dict,
        convert_expr,
    )

    timings = {}
    de = _get_diffengine()

    t0 = time.perf_counter()
    variables = problem.variables()
    var_dict, n_vars = build_variable_dict(variables)
    timings['build_variable_dict'] = time.perf_counter() - t0

    # Convert objective expression
    t0 = time.perf_counter()
    c_obj = convert_expr(problem.objective.expr, var_dict, n_vars)
    timings['convert_expr(objective)'] = time.perf_counter() - t0

    # Convert constraint argument expressions
    t0 = time.perf_counter()
    expr_list = [arg for c in ordered_cons for arg in c.args]
    c_constraints = [convert_expr(e, var_dict, n_vars) for e in expr_list]
    timings['convert_expr(constraints)'] = time.perf_counter() - t0

    # Build the diff engine problem
    t0 = time.perf_counter()
    capsule = de.make_problem(c_obj, c_constraints)
    timings['de.make_problem'] = time.perf_counter() - t0

    # Create flattened variable with MIP info
    boolean, integer = extract_mip_idx(variables)
    x = Variable(n_vars, boolean=boolean, integer=integer)

    # Evaluate at x0 = 0
    x0 = np.zeros(n_vars, dtype=np.float64)

    # --- Objective ---
    t0 = time.perf_counter()
    if quad_obj:
        de.problem_init_derivatives(capsule)   # Init both Jacobian + Hessian
    else:
        de.problem_init_jacobian(capsule)      # Init Jacobian only (skip Hessian)
    timings['de.problem_init_derivatives'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    d = float(de.problem_objective_forward(capsule, x0))
    timings['de.problem_objective_forward'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    q = de.problem_gradient(capsule).copy()
    timings['de.problem_gradient'] = time.perf_counter() - t0

    # --- Constraints ---
    if c_constraints:
        t0 = time.perf_counter()
        b_vec = de.problem_constraint_forward(capsule, x0)
        timings['de.problem_constraint_forward'] = time.perf_counter() - t0

        # Get Jacobian as CSR components — keep as CSR to avoid costly conversion.
        # Downstream consumers (e.g. osqp_qpif) convert to CSC themselves.
        t0 = time.perf_counter()
        jac_data, jac_indices, jac_indptr, jac_shape = de.problem_jacobian(capsule)
        timings['de.problem_jacobian'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = jac_shape[0]
        A = sp.csr_matrix((jac_data, jac_indices, jac_indptr), shape=(m, n_vars))
        timings['build_csr_matrix'] = time.perf_counter() - t0
    else:
        b_vec = np.array([], dtype=np.float64)
        A = sp.csr_matrix((0, n_vars))

    # --- Quadratic objective (Hessian) ---
    P = None
    if quad_obj:
        t0 = time.perf_counter()
        duals = np.zeros(b_vec.shape[0], dtype=np.float64)
        h_data, h_indices, h_indptr, h_shape = de.problem_hessian(capsule, 1.0, duals)
        timings['de.problem_hessian'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        P_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
        P = P_csr + P_csr.T - sp.diags(P_csr.diagonal())
        P = sp.csc_matrix(P)
        timings['hessian_symmetrize'] = time.perf_counter() - t0

    # Extract bounds from original variables
    t0 = time.perf_counter()
    from cvxpy.reductions.matrix_stuffing import extract_lower_bounds, extract_upper_bounds
    lower_bounds = extract_lower_bounds(variables, n_vars)
    upper_bounds = extract_upper_bounds(variables, n_vars)
    timings['extract_bounds'] = time.perf_counter() - t0

    # Log all timings
    s_settings.LOGGER.info('[build_diffengine_cone_program] Timing breakdown:')
    total = 0.0
    for label, elapsed in timings.items():
        s_settings.LOGGER.info('  %-40s %.4f s', label, elapsed)
        total += elapsed
    s_settings.LOGGER.info('  %-40s %.4f s', 'TOTAL (instrumented)', total)

    return DiffengineConeProgram(
        x=x,
        A=A,
        b=np.atleast_1d(b_vec),
        q=q,
        d=d,
        P=P,
        constraints=ordered_cons,
        variables=variables,
        var_id_to_col=inverse_data.var_offsets,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

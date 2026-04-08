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

import numpy as np
import scipy.sparse as sp
from sparsediffpy import _sparsediffengine as _diffengine

from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims, ParamConeProg
from cvxpy.reductions.matrix_stuffing import (
    extract_lower_bounds,
    extract_mip_idx,
    extract_upper_bounds,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
    build_capsule,
)
from cvxpy.reductions.utilities import group_constraints


class DiffengineConeProgram(ParamConeProg):
    """A cone program with matrices extracted via the diff engine.

    Duck-type compatible with ParamConeProg. On first solve, stores concrete
    A, b, q, d, P matrices. When parameters are present, re-evaluates the
    C expression DAG on subsequent solves via apply_parameters().

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
        inverse_data,
        formatted: bool = False,
        lower_bounds=None,
        upper_bounds=None,
        capsule=None,
        parameters=None,
        quad_obj: bool = False,
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

        self._inverse_data = inverse_data
        self.variables = [inverse_data.id2var[vid] for vid in inverse_data.var_offsets]
        self.var_id_to_col = inverse_data.var_offsets
        self.id_to_var = inverse_data.id2var

        self.formatted = formatted
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Parameter support
        self._capsule = capsule
        self._quad_obj = quad_obj
        self._restruct_mat = None  # None = no restruct, False = identity (skip)

        self.parameters = list(parameters) if parameters else []
        self.param_id_to_col = inverse_data.param_id_map
        self.id_to_param = {p.id: p for p in self.parameters}
        self.param_id_to_size = inverse_data.param_to_size
        self.total_param_size = sum(p.size for p in self.parameters)

        # No parametric bound tensors.
        self.lb_tensor = None
        self.ub_tensor = None

    def is_mixed_integer(self) -> bool:
        return self.x.attributes['boolean'] or self.x.attributes['integer']

    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False, quad_obj: bool = False):
        """Return problem matrices, re-evaluating if parameters are present.

        When no parameters exist, returns the stored matrices directly.
        When parameters exist, updates the C DAG with current parameter values
        and re-evaluates objective/constraints at x=0.
        """
        if self.total_param_size == 0 or self._capsule is None:
            # Non-parametric: return stored matrices directly.
            A = self._A
            if quad_obj and self.P is not None:
                return self.P, self._q, self._d, A, self._b
            return self._q, self._d, A, self._b

        # Build theta vector from current parameter values.
        if id_to_param_value is not None:
            parts = [np.asarray(id_to_param_value[p.id],
                                dtype=np.float64).flatten(order='F')
                     for p in self.parameters]
        else:
            parts = [np.asarray(p.value,
                                dtype=np.float64).flatten(order='F')
                     for p in self.parameters]
        theta = np.concatenate(parts)

        _diffengine.problem_update_params(self._capsule, theta)

        # Re-evaluate at x0 = 0.
        n_vars = self._inverse_data.x_length
        x0 = np.zeros(n_vars, dtype=np.float64)

        d = float(_diffengine.problem_objective_forward(self._capsule, x0))
        q = _diffengine.problem_gradient(self._capsule).copy()

        if self.constr_size > 0:
            b_vec = _diffengine.problem_constraint_forward(self._capsule, x0)
            jac_data, jac_indices, jac_indptr, jac_shape = \
                _diffengine.problem_jacobian(self._capsule)
            A = sp.csr_matrix(
                (jac_data, jac_indices, jac_indptr),
                shape=(jac_shape[0], n_vars))
        else:
            b_vec = np.array([], dtype=np.float64)
            A = sp.csr_matrix((0, n_vars))

        b = np.atleast_1d(b_vec)

        # Apply cached restructuring matrix if present.
        if self._restruct_mat is not None and self._restruct_mat is not False:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()

        # Update stored matrices.
        self._A, self._b, self._q, self._d = A, b, q, d

        if quad_obj:
            duals = np.zeros(b.shape[0], dtype=np.float64)
            h_data, h_indices, h_indptr, h_shape = \
                _diffengine.problem_hessian(self._capsule, 1.0, duals)
            P_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
            self.P = sp.csc_matrix(P_csr + P_csr.T - sp.diags(P_csr.diagonal()))
            return self.P, q, d, A, b
        return q, d, A, b

    def apply_restruct_mat(self, restruct_mat, restruct_mat_op=None):
        """Apply restructuring matrix to concrete A, b matrices.

        Materializes the block-diagonal restructuring matrix R and caches it
        so that parametric re-evaluations can re-apply it.

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
        R = None
        if restruct_mat:
            sparse_mats = []
            for mat in restruct_mat:
                if sp.issparse(mat):
                    sparse_mats.append(sp.csc_matrix(mat))
                elif callable(mat):
                    eye = sp.eye_array(mat.shape[1], format='csc')
                    sparse_mats.append(sp.csc_matrix(mat(eye)))
                else:
                    eye = sp.eye_array(mat.shape[1], format='csc')
                    sparse_mats.append(sp.csc_matrix(mat @ eye))

            R = sp.block_diag(sparse_mats, format='csc')
            new_A = R @ self._A
            new_b = np.asarray(R @ self._b).flatten()
        else:
            new_A, new_b = self._A, self._b

        new_prog = DiffengineConeProgram(
            self.x, new_A, new_b, self._q, self._d, self.P,
            self.constraints, self._inverse_data,
            formatted=True,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            capsule=self._capsule,
            parameters=self.parameters,
            quad_obj=self._quad_obj,
        )
        # Detect identity R to skip R @ A on re-solves.
        if R is not None:
            is_identity = (R.shape[0] == R.shape[1]
                           and R.nnz == R.shape[0]
                           and np.allclose(R.data, 1.0))
            new_prog._restruct_mat = False if is_identity else R
        return new_prog

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
    Supports CVXPY Parameters via the C engine's parameter nodes.

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
    # Build the diff engine problem capsule.
    expr_list = [arg for c in ordered_cons for arg in c.args]
    params = problem.parameters()
    capsule, n_vars, _ = build_capsule(
        problem.objective.expr, expr_list, inverse_data, params=params)

    # Create flattened variable with MIP info.
    boolean, integer = extract_mip_idx(problem.variables())
    x = Variable(n_vars, boolean=boolean, integer=integer)

    # Evaluate at x0 = 0.
    x0 = np.zeros(n_vars, dtype=np.float64)

    # --- Initialize derivatives ---
    if quad_obj:
        _diffengine.problem_init_derivatives(capsule)
    else:
        _diffengine.problem_init_jacobian(capsule)

    # --- Objective ---
    d = float(_diffengine.problem_objective_forward(capsule, x0))
    q = _diffengine.problem_gradient(capsule).copy()

    # --- Constraints ---
    if expr_list:
        b_vec = _diffengine.problem_constraint_forward(capsule, x0)
        jac_data, jac_indices, jac_indptr, jac_shape = \
            _diffengine.problem_jacobian(capsule)
        m = jac_shape[0]
        A = sp.csr_matrix((jac_data, jac_indices, jac_indptr), shape=(m, n_vars))
    else:
        b_vec = np.array([], dtype=np.float64)
        A = sp.csr_matrix((0, n_vars))

    # --- Quadratic objective (Hessian) ---
    P = None
    if quad_obj:
        duals = np.zeros(b_vec.shape[0], dtype=np.float64)
        h_data, h_indices, h_indptr, h_shape = \
            _diffengine.problem_hessian(capsule, 1.0, duals)
        P_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
        P = sp.csc_matrix(P_csr + P_csr.T - sp.diags(P_csr.diagonal()))

    # Extract bounds from original variables.
    lower_bounds = extract_lower_bounds(problem.variables(), n_vars)
    upper_bounds = extract_upper_bounds(problem.variables(), n_vars)

    return DiffengineConeProgram(
        x=x,
        A=A,
        b=np.atleast_1d(b_vec),
        q=q,
        d=d,
        P=P,
        constraints=ordered_cons,
        inverse_data=inverse_data,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        capsule=capsule,
        parameters=params,
        quad_obj=quad_obj,
    )

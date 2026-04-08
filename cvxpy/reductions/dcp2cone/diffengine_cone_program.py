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
        variables: list,
        var_id_to_col: dict,
        formatted: bool = False,
        lower_bounds=None,
        upper_bounds=None,
        # Parameter support fields
        capsule=None,
        parameters=None,
        param_id_to_col=None,
        quad_obj: bool = False,
        n_vars: int = 0,
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

        # Parameter support
        self._capsule = capsule
        self._quad_obj = quad_obj
        self._n_vars = n_vars
        self._restruct_mat = None  # None = no restruct, False = identity (skip)

        # Cached arrays for fast re-evaluation (populated on first apply_parameters)
        self._x0 = np.zeros(n_vars, dtype=np.float64) if n_vars > 0 else None
        self._hess_sym_template = None  # cached CSC with symmetric sparsity pattern

        if parameters:
            self.parameters = list(parameters)
            self.param_id_to_col = dict(param_id_to_col) if param_id_to_col else {}
            self.id_to_param = {p.id: p for p in self.parameters}
            self.param_id_to_size = {p.id: p.size for p in self.parameters}
            self.total_param_size = sum(p.size for p in self.parameters)
        else:
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

        de = _get_diffengine()

        # Build theta vector from current parameter values.
        if id_to_param_value is not None:
            parts = [np.asarray(np.array(id_to_param_value[p.id]),
                                dtype=np.float64).flatten(order='C')
                     for p in self.parameters]
        else:
            parts = [np.asarray(np.array(p.value),
                                dtype=np.float64).flatten(order='C')
                     for p in self.parameters]
        theta = np.concatenate(parts)

        de.problem_update_params(self._capsule, theta)

        # Re-evaluate at x0 = 0 (cached).
        x0 = self._x0

        d = float(de.problem_objective_forward(self._capsule, x0))
        q = de.problem_gradient(self._capsule).copy()

        if self.constr_size > 0:
            b_vec = de.problem_constraint_forward(self._capsule, x0)
            jac_data, jac_indices, jac_indptr, jac_shape = \
                de.problem_jacobian(self._capsule)
            A = sp.csr_matrix(
                (jac_data, jac_indices, jac_indptr),
                shape=(jac_shape[0], self._n_vars))
        else:
            b_vec = np.array([], dtype=np.float64)
            A = sp.csr_matrix((0, self._n_vars))

        P = None
        if quad_obj:
            duals = np.zeros(b_vec.shape[0], dtype=np.float64)
            h_data, h_indices, h_indptr, h_shape = \
                de.problem_hessian(self._capsule, 1.0, duals)
            P = self._symmetrize_hessian(h_data, h_indices, h_indptr, h_shape)

        b = np.atleast_1d(b_vec)

        # Apply cached restructuring matrix if present.
        # _restruct_mat is None (not set), False (identity, skip), or a sparse matrix.
        if self._restruct_mat is not None and self._restruct_mat is not False:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()

        # Update stored matrices.
        self._A, self._b, self._q, self._d = A, b, q, d
        if P is not None:
            self.P = P

        if quad_obj and self.P is not None:
            return self.P, q, d, A, b
        return q, d, A, b

    def _symmetrize_hessian(self, h_data, h_indices, h_indptr, h_shape):
        """Symmetrize a lower-triangular Hessian returned by the C engine.

        On first call, builds and caches the symmetric CSC sparsity pattern.
        On subsequent calls, only updates the data array (O(nnz), no scipy overhead).
        """
        P_lower = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)

        if self._hess_sym_template is None:
            # First call: build the symmetric pattern and cache it.
            r, c = P_lower.nonzero()
            v = np.asarray(P_lower[r, c]).ravel()
            mask = r != c
            all_r = np.concatenate([r, c[mask]])
            all_c = np.concatenate([c, r[mask]])
            all_v = np.concatenate([v, v[mask]])
            P_sym = sp.csc_matrix((all_v, (all_r, all_c)), shape=h_shape)
            P_sym.sort_indices()
            # Cache the template and the index mapping for fast updates.
            self._hess_sym_template = P_sym
            self._hess_lower_rows = r
            self._hess_lower_cols = c
            self._hess_offdiag_mask = mask
            return P_sym

        # Fast path: reuse cached pattern, just update values.
        r = self._hess_lower_rows
        c = self._hess_lower_cols
        mask = self._hess_offdiag_mask
        v = np.asarray(P_lower[r, c]).ravel()
        all_v = np.concatenate([v, v[mask]])
        P = self._hess_sym_template.copy()
        P.data[:] = all_v
        return P

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
            self.constraints, self.variables, self.var_id_to_col,
            formatted=True,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            capsule=self._capsule,
            parameters=self.parameters,
            param_id_to_col=self.param_id_to_col,
            quad_obj=self._quad_obj,
            n_vars=self._n_vars,
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
    from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import convert_expr
    from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
        build_param_dict,
        build_var_dict,
    )

    de = _get_diffengine()

    # Build variable and parameter dictionaries.
    var_dict, n_vars = build_var_dict(inverse_data)
    param_dict = build_param_dict(inverse_data)

    # Convert objective expression.
    c_obj = convert_expr(problem.objective.expr, var_dict, n_vars, param_dict)

    # Convert constraint argument expressions.
    expr_list = [arg for c in ordered_cons for arg in c.args]
    c_constraints = [convert_expr(e, var_dict, n_vars, param_dict) for e in expr_list]

    # Build the diff engine problem.
    capsule = de.make_problem(c_obj, c_constraints)

    # Register and initialize parameters if present.
    params = problem.parameters()
    if param_dict:
        de.problem_register_params(capsule, list(param_dict.values()))
        theta = np.concatenate([
            np.asarray(p.value, dtype=np.float64).flatten(order='C')
            for p in params
        ])
        de.problem_update_params(capsule, theta)

    # Create flattened variable with MIP info.
    variables = problem.variables()
    boolean, integer = extract_mip_idx(variables)
    x = Variable(n_vars, boolean=boolean, integer=integer)

    # Evaluate at x0 = 0.
    x0 = np.zeros(n_vars, dtype=np.float64)

    # --- Initialize derivatives ---
    if quad_obj:
        de.problem_init_derivatives(capsule)
    else:
        de.problem_init_jacobian(capsule)

    # --- Objective ---
    d = float(de.problem_objective_forward(capsule, x0))
    q = de.problem_gradient(capsule).copy()

    # --- Constraints ---
    if c_constraints:
        b_vec = de.problem_constraint_forward(capsule, x0)
        jac_data, jac_indices, jac_indptr, jac_shape = de.problem_jacobian(capsule)
        m = jac_shape[0]
        A = sp.csr_matrix((jac_data, jac_indices, jac_indptr), shape=(m, n_vars))
    else:
        b_vec = np.array([], dtype=np.float64)
        A = sp.csr_matrix((0, n_vars))

    # --- Quadratic objective (Hessian) ---
    P = None
    if quad_obj:
        duals = np.zeros(b_vec.shape[0], dtype=np.float64)
        h_data, h_indices, h_indptr, h_shape = de.problem_hessian(capsule, 1.0, duals)
        P_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
        P = P_csr + P_csr.T - sp.diags(P_csr.diagonal())
        P = sp.csc_matrix(P)

    # Extract bounds from original variables.
    from cvxpy.reductions.matrix_stuffing import extract_lower_bounds, extract_upper_bounds
    lower_bounds = extract_lower_bounds(variables, n_vars)
    upper_bounds = extract_upper_bounds(variables, n_vars)

    # Build parameter metadata for DPP caching.
    param_id_to_col = {}
    if params:
        offset = 0
        for p in params:
            param_id_to_col[p.id] = offset
            offset += p.size

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
        capsule=capsule,
        parameters=params,
        param_id_to_col=param_id_to_col,
        quad_obj=quad_obj,
        n_vars=n_vars,
    )

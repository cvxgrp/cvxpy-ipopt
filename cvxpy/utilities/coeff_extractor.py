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

import operator
from typing import List

import numpy as np
import scipy.sparse as sp

from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.lin_ops.lin_op import NO_OP, LinOp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.replace_quad_forms import (
    replace_quad_forms,
    restore_quad_forms,
)


# TODO find best format for sparse matrices: csr, csc, dok, lil, ...
class CoeffExtractor:

    def __init__(self, inverse_data, canon_backend: str | None) -> None:
        self.id_map = inverse_data.var_offsets
        self.x_length = inverse_data.x_length
        self.var_shapes = inverse_data.var_shapes
        self.param_shapes = inverse_data.param_shapes
        self.param_to_size = inverse_data.param_to_size
        self.param_id_map = inverse_data.param_id_map
        self.canon_backend = canon_backend

    def affine(self, expr):
        """Extract problem data tensor from an expression that is reducible to
        A*x + b.

        Applying the tensor to a flattened parameter vector and reshaping
        will recover A and b (see the helpers in canonInterface).

        Parameters
        ----------
        expr : Expression or list of Expressions.
            The expression(s) to process.

        Returns
        -------
        SciPy CSR matrix
            Problem data tensor, of shape
            (constraint length * (variable length + 1), parameter length + 1)
        """
        if isinstance(expr, list):
            expr_list = expr
        else:
            expr_list = [expr]
        assert all([e.is_dpp() for e in expr_list])
        num_rows = sum([e.size for e in expr_list])

        # Check for DIFFENGINE backend
        import cvxpy.settings as s
        if self.canon_backend == s.DIFFENGINE_CANON_BACKEND:
            return self._affine_diffengine(expr_list, num_rows)

        # Existing path for other backends
        op_list = [e.canonical_form[0] for e in expr_list]
        return canonInterface.get_problem_matrix(op_list,
                                                 self.x_length,
                                                 self.id_map,
                                                 self.param_to_size,
                                                 self.param_id_map,
                                                 num_rows,
                                                 self.canon_backend)

    def _affine_diffengine(self, expr_list, num_rows):
        """Extract coefficients using the C-based diff engine.

        Analogous to existing backend path which does:
            op_list = [e.canonical_form[0] for e in expr_list]
            return canonInterface.get_problem_matrix(op_list, ...)

        Instead, we use the diff engine's jacobian computation directly.
        For affine expressions f(x) = Ax + b, the Jacobian equals A and f(0) = b.

        Parameters
        ----------
        expr_list : list of Expression
            List of CVXPY expressions to extract coefficients from.
        num_rows : int
            Total number of rows (sum of expression sizes).

        Returns
        -------
        SciPy CSC matrix
            Problem data tensor in same format as other backends.
        """
        try:
            from dnlp_diff_engine import _core as _diffengine
            from dnlp_diff_engine import _convert_expr
        except ImportError:
            import warnings
            import cvxpy.settings as s
            warnings.warn(
                "dnlp_diff_engine not installed. Falling back to SCIPY backend.",
                stacklevel=3
            )
            self.canon_backend = s.SCIPY_CANON_BACKEND
            op_list = [e.canonical_form[0] for e in expr_list]
            return canonInterface.get_problem_matrix(
                op_list,
                self.x_length,
                self.id_map,
                self.param_to_size,
                self.param_id_map,
                num_rows,
                self.canon_backend
            )

        # Collect all variables referenced in expr_list
        expr_vars = set()
        for expr in expr_list:
            expr_vars.update(expr.variables())

        # Check if expressions reference all variables in self.id_map
        expr_var_ids = {v.id for v in expr_vars}
        expected_var_ids = set(self.id_map.keys())

        if expr_var_ids != expected_var_ids:
            # Variables don't match - fall back to SCIPY backend
            import cvxpy.settings as s
            op_list = [e.canonical_form[0] for e in expr_list]
            return canonInterface.get_problem_matrix(
                op_list,
                self.x_length,
                self.id_map,
                self.param_to_size,
                self.param_id_map,
                num_rows,
                s.SCIPY_CANON_BACKEND
            )

        try:
            # Build variable dict using self.id_map ordering (crucial for correctness!)
            # self.id_map maps var_id -> offset, which defines the variable ordering
            var_dict = {}
            for var in expr_vars:
                offset = self.id_map[var.id]
                shape = self.var_shapes[var.id]
                if len(shape) == 2:
                    d1, d2 = shape[0], shape[1]
                elif len(shape) == 1:
                    d1, d2 = shape[0], 1
                else:  # scalar
                    d1, d2 = 1, 1
                c_var = _diffengine.make_variable(d1, d2, offset, self.x_length)
                var_dict[var.id] = c_var

            # Convert expressions to C expressions and stack as constraints
            c_constraints = []
            for expr in expr_list:
                c_expr = _convert_expr(expr, var_dict, self.x_length)
                c_constraints.append(c_expr)

            # Create a C problem with dummy objective (constant 0)
            c_objective = _diffengine.make_constant(1, 1, self.x_length, np.array([0.0]))
            c_prob = _diffengine.make_problem(c_objective, c_constraints)
            _diffengine.problem_init_jacobian_only(c_prob)

            # Evaluate at x=0 to get constant offset b
            x_zero = np.zeros(self.x_length)
            b = _diffengine.problem_constraint_forward(c_prob, x_zero)

            # Get Jacobian (= A for affine expressions)
            jac_data = _diffengine.problem_jacobian(c_prob)
            from scipy import sparse
            A = sparse.csr_matrix(jac_data[:3], shape=jac_data[3])

            # Reshape to match expected tensor format
            return self._reshape_to_tensor_format(A, b, num_rows)

        except Exception as e:
            # Fall back to SCIPY backend if diff engine fails
            import warnings
            import cvxpy.settings as s
            warnings.warn(
                f"Diff engine failed ({e}). Falling back to SCIPY backend.",
                stacklevel=3
            )
            self.canon_backend = s.SCIPY_CANON_BACKEND
            op_list = [expr.canonical_form[0] for expr in expr_list]
            return canonInterface.get_problem_matrix(
                op_list,
                self.x_length,
                self.id_map,
                self.param_to_size,
                self.param_id_map,
                num_rows,
                self.canon_backend
            )

    def _reshape_to_tensor_format(self, A, b, num_rows):
        """Reshape A matrix and b vector to match canon backend output format.

        The canon backend returns a sparse matrix of shape:
            (num_rows * (x_length + 1), param_size + 1)

        For non-parametrized problems (param_size + 1 = 1), this is:
            (num_rows * (x_length + 1), 1)

        which represents [A | b] stacked in Fortran (column-major) order.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Coefficient matrix of shape (num_rows, x_length).
        b : np.ndarray
            Constant offset vector of shape (num_rows,).
        num_rows : int
            Number of constraint rows.

        Returns
        -------
        scipy.sparse.csc_matrix
            Reshaped tensor matching canon backend format.
        """
        # Stack A and b horizontally: [A | b] with shape (num_rows, x_length + 1)
        b_col = sp.csc_matrix(b.reshape(-1, 1))
        Ab = sp.hstack([A, b_col], format='csc')

        # Flatten in Fortran order to match canon backend
        # Result shape: (num_rows * (x_length + 1), 1)
        data = Ab.toarray().flatten(order='F')
        return sp.csc_matrix(data.reshape(-1, 1))

    def extract_quadratic_coeffs(self, affine_expr, quad_forms):
        """ Assumes quadratic forms all have variable arguments.
            Affine expressions can be anything.
        """
        assert affine_expr.is_dpp()
        # Here we take the problem objective, replace all the SymbolicQuadForm
        # atoms with variables of the same dimensions.
        # We then apply the canonInterface to reduce the "affine head"
        # of the expression tree to a coefficient vector c and constant offset d.
        # Because the expression is parameterized, we extend that to a matrix
        # [c1 c2 ...]
        # [d1 d2 ...]
        # where ci,di are the vector and constant for the ith parameter.
        affine_id_map, affine_offsets, x_length, affine_var_shapes = \
            InverseData.get_var_offsets(affine_expr.variables())
        op_list = [affine_expr.canonical_form[0]]

        # DIFFENGINE backend only supports affine() method, not quadratic extraction
        # Fall back to SCIPY for quadratic coefficient extraction
        import cvxpy.settings as s
        backend = (s.SCIPY_CANON_BACKEND if self.canon_backend == s.DIFFENGINE_CANON_BACKEND
                   else self.canon_backend)

        param_coeffs = canonInterface.get_problem_matrix(op_list,
                                                         x_length,
                                                         affine_offsets,
                                                         self.param_to_size,
                                                         self.param_id_map,
                                                         affine_expr.size,
                                                         backend)

        # Iterates over every entry of the parameters vector,
        # and obtains the Pi and qi for that entry i.
        # These are then combined into matrices [P1.flatten(), P2.flatten(), ...]
        # and [q1, q2, ...]
        constant = param_coeffs[[-1], :]
        # TODO keep sparse.
        c = param_coeffs[:-1, :].toarray()
        num_params = param_coeffs.shape[1]

        # coeffs stores the P and q for each quad_form,
        # as well as for true variable nodes in the objective.
        coeffs = {}
        # The goal of this loop is to appropriately multiply
        # the matrix P of each quadratic term by the coefficients
        # in param_coeffs. Later we combine all the quadratic terms
        # to form a single matrix P.
        for var in affine_expr.variables():
            # quad_forms maps the ids of the SymbolicQuadForm atoms
            # in the objective to (modified parent node of quad form,
            #                      argument index of quad form,
            #                      quad form atom)
            if var.id in quad_forms:
                # This was a dummy variable
                var_id = var.id
                orig_id = quad_forms[var_id][2].args[0].id
                var_offset = affine_id_map[var_id][0]
                var_size = affine_id_map[var_id][1]
                c_part = c[var_offset:var_offset+var_size, :]

                # Convert to sparse matrix.
                quad_form_atom = quad_forms[var_id][2]
                P = quad_form_atom.P
                assert (
                    P.value is not None
                ), "P matrix must be instantiated before calling extract_quadratic_coeffs."
                if sp.issparse(P) and not isinstance(P, sp.coo_matrix):
                    P = P.value.tocoo()
                else:
                    P = sp.coo_matrix(P.value)

                # Get block structure if available
                block_indices = quad_form_atom.block_indices

                # We multiply P by the parameter coefficients.
                if var_size == 1:
                    # SCALAR PATH - Single quad form in the expression, i.e.,
                    # we multiply the full P matrix by the non-zero entries of c_part.
                    nonzero_idxs = c_part[0] != 0
                    data = P.data[:, None] * c_part[:, nonzero_idxs]
                    param_idxs = np.arange(num_params)[nonzero_idxs]
                    P_tup = TensorRepresentation(
                        data.flatten(order="F"),
                        np.tile(P.row, len(param_idxs)),
                        np.tile(P.col, len(param_idxs)),
                        np.repeat(param_idxs, len(P.data)),
                        P.shape
                    )
                elif block_indices is not None:
                    # BLOCK-STRUCTURED PATH - Non-scalar output with block structure.
                    # Each output element j depends on input indices block_indices[j].
                    P_tup = self._extract_block_quad(P, c_part, block_indices, num_params)
                else:
                    # DIAGONAL PATH - Multiple quad forms in the one expression,
                    # i.e., c_part is now a matrix where each row corresponds to
                    # a different variable.
                    assert (P.col == P.row).all(), \
                        "Only diagonal P matrices are supported for multiple quad forms " \
                        "without block_indices. If you need non-diagonal structure, " \
                        "use SymbolicQuadForm with block_indices parameter."

                    scaled_c_part = P @ c_part
                    paramx_idx_row, param_idx_col = np.nonzero(scaled_c_part)
                    c_vals = c_part[paramx_idx_row, param_idx_col]
                    P_tup = TensorRepresentation(
                        c_vals,
                        paramx_idx_row,
                        paramx_idx_row.copy(),
                        param_idx_col,
                        P.shape
                    )

                if orig_id in coeffs:
                    if 'P' in coeffs[orig_id]:
                        coeffs[orig_id]['P'] =  coeffs[orig_id]['P'] + P_tup
                    else:
                        coeffs[orig_id]['P'] = P_tup
                else:
                    # No q for dummy variables.
                    coeffs[orig_id] = dict()
                    coeffs[orig_id]['P'] = P_tup
                    shape = (P.shape[0], c.shape[1])
                    if num_params == 1:
                        # Fast path for no parameters, keep q dense.
                        coeffs[orig_id]['q'] = np.zeros(shape)
                    else:
                        coeffs[orig_id]['q'] = sp.coo_matrix(([], ([], [])), shape=shape) 
            else:
                # This was a true variable, so it can only have a q term.
                var_offset = affine_id_map[var.id][0]
                var_size = np.prod(affine_var_shapes[var.id], dtype=int)
                if var.id in coeffs:
                    # Fast path for no parameters, q is dense and so is c.
                    if num_params == 1:
                        coeffs[var.id]['q'] += c[var_offset:var_offset+var_size, :]
                    else:
                        coeffs[var.id]['q'] += param_coeffs[var_offset:var_offset+var_size, :]
                else:   
                    coeffs[var.id] = dict()
                    # Fast path for no parameters, q is dense and so is c.
                    if num_params == 1:
                        coeffs[var.id]['q'] = c[var_offset:var_offset+var_size, :]
                    else:
                        coeffs[var.id]['q'] = param_coeffs[var_offset:var_offset+var_size, :]
        return coeffs, constant

    def _extract_block_quad(
        self,
        P: sp.coo_matrix,
        c_part: np.ndarray,
        block_indices: List[np.ndarray],
        num_params: int,
    ) -> TensorRepresentation:
        """Extract quadratic coefficients for block-structured quad forms.

        Each output element j uses input indices from block_indices[j].
        Supports both contiguous and non-contiguous blocks.

        Args:
            P: COO sparse matrix (N x N)
            c_part: Coefficients (num_outputs x num_params)
            block_indices: List of np.ndarray, each containing indices for that block
            num_params: Number of parameter columns

        Returns:
            TensorRepresentation for the scaled P matrix
        """
        all_data = []
        all_row = []
        all_col = []
        all_param = []

        for j, indices in enumerate(block_indices):
            # Filter P entries where both row and col are in this block
            row_mask = np.isin(P.row, indices)
            col_mask = np.isin(P.col, indices)
            mask = row_mask & col_mask

            if not mask.any():
                continue

            block_data = P.data[mask]
            block_row = P.row[mask]
            block_col = P.col[mask]

            # Coefficient for this output element
            coef_row = c_part[j, :]
            nonzero_params = np.nonzero(coef_row)[0]

            if len(nonzero_params) == 0:
                continue

            # Scale by each non-zero coefficient
            for param_idx in nonzero_params:
                scaled_data = block_data * coef_row[param_idx]
                all_data.append(scaled_data)
                all_row.append(block_row)  # Already global coordinates
                all_col.append(block_col)
                all_param.append(np.full(len(scaled_data), param_idx, dtype=int))

        if not all_data:
            return TensorRepresentation.empty_with_shape(P.shape)

        return TensorRepresentation(
            np.concatenate(all_data),
            np.concatenate(all_row),
            np.concatenate(all_col),
            np.concatenate(all_param),
            P.shape,
        )

    def quad_form(self, expr):
        """Extract quadratic, linear constant parts of a quadratic objective.
        """
        # Insert no-op such that root is never a quadratic form, for easier
        # processing
        root = LinOp(NO_OP, expr.shape, [expr], [])

        # Replace quadratic forms with dummy variables.
        quad_forms = replace_quad_forms(root, {})

        # Calculate affine parts and combine them with quadratic forms to get
        # the coefficients.
        coeffs, constant = self.extract_quadratic_coeffs(root.args[0],
                                                         quad_forms)
        # Restore expression.
        restore_quad_forms(root.args[0], quad_forms)

        # Sort variables corresponding to their starting indices, in ascending
        # order.
        offsets = sorted(self.id_map.items(), key=operator.itemgetter(1))

        # Extract quadratic matrices and vectors
        num_params = constant.shape[1]
        P_list = []
        q_list = []
        P_height = 0
        P_entries = 0
        for var_id, _ in offsets:
            shape = self.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            if var_id in coeffs and 'P' in coeffs[var_id]:
                P = coeffs[var_id]['P']
                P_entries += P.data.size
            else:
                P = TensorRepresentation.empty_with_shape((size, size))
            if var_id in coeffs and 'q' in coeffs[var_id]:
                q = coeffs[var_id]['q']
            else:
                # Fast path for no parameters.
                if num_params == 1:
                    q = np.zeros((size, num_params))
                else:
                    q = sp.coo_matrix(([], ([], [])), (size, num_params))

            P_list.append(P)
            q_list.append(q)
            P_height += size

        if P_height != self.x_length:
            raise ValueError("Resulting quadratic form does not have "
                               "appropriate dimensions")

        # Stitch together Ps and qs and constant.
        P = self.merge_P_list(P_list, P_height, num_params)
        q = self.merge_q_list(q_list, constant, num_params)
        return P, q

    def merge_P_list(
            self, 
            P_list: List[TensorRepresentation],
            P_height: int, 
            num_params: int,
        ) -> sp.csc_array:
        """Conceptually we build a block diagonal matrix
           out of all the Ps, then flatten the first two dimensions.
           eg P1
                P2
           We do this by extending each P with zero blocks above and below.

        Args:
            P_list: list of P submatrices as TensorRepresentation objects.
            P_entries: number of entries in the merged P matrix.
            P_height: number of rows in the merged P matrix.
            num_params: number of parameters in the problem.
        
        Returns:
            A CSC sparse representation of the merged P matrix.
        """

        offset = 0
        for P in P_list:
            m, n = P.shape
            assert m == n
            assert P.row is not P.col

            # Translate local to global indices within the block diagonal matrix.
            P.row += offset
            P.col += offset
            P.shape = (P_height, P_height)
    
            offset += m

        combined = TensorRepresentation.combine(P_list)

        return combined.flatten_tensor(num_params)

    def merge_q_list(
        self,
        q_list: List[sp.spmatrix | np.ndarray],
        constant: sp.csc_array,
        num_params: int,
    ) -> sp.csr_array:
        """Stack q with constant offset as last row.

        Args:
            q_list: list of q submatrices as SciPy sparse matrices or NumPy arrays.
            constant: The constant offset as a CSC sparse matrix.
            num_params: number of parameters in the problem.

        Returns:
            A CSR sparse representation of the merged q matrix.
        """
        # Fast path for no parameters.
        if num_params == 1:
            q = np.vstack(q_list)
            q = np.vstack([q, constant.toarray()])
            return sp.csr_array(q)
        else:
            q = sp.vstack(q_list + [constant])
            return sp.csr_array(q)

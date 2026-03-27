
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

Converters from CVXPY expressions to C diff engine expressions.

This module provides the mapping between CVXPY atom types and their
corresponding SparseDiffPy constructors.
"""
import numpy as np
from scipy import sparse
from sparsediffpy import _sparsediffengine as _diffengine

import cvxpy as cp
from cvxpy.expressions.constants.parameter import Parameter


def build_theta(parameters):
    """Build flat theta vector by concatenating current parameter values.

    Args:
        parameters: list of cp.Parameter objects

    Returns:
        theta: 1D numpy array of parameter values
    """
    return np.concatenate([
        np.asarray(p.value, dtype=np.float64).flatten(order='F')
        for p in parameters
    ])


def normalize_shape(shape):
    """Normalize shape to 2D (d1, d2) for the C engine."""
    shape = tuple(shape)
    return (1,) * (2 - len(shape)) + shape


class ConvertContext:
    """State for converting CVXPY expressions to C diff engine expressions.

    Builds C variable and parameter node dicts from InverseData.
    """

    def __init__(self, inverse_data):
        self.n_vars = inverse_data.x_length
        self.var_dict = self._build_var_dict(inverse_data)
        self.param_dict = self._build_param_dict(inverse_data)

    def _build_var_dict(self, inv):
        """Build {var_id: C variable capsule} mapping."""
        var_dict = {}
        for var_id, (offset, _) in inv.id_map.items():
            d1, d2 = normalize_shape(inv.var_shapes[var_id])
            var_dict[var_id] = _diffengine.make_variable(
                d1, d2, offset, self.n_vars)
        return var_dict

    def _build_param_dict(self, inv):
        """Build {param_id: C parameter capsule} mapping."""
        param_dict = {}
        for param_id, offset in inv.param_id_map.items():
            if param_id not in inv.param_shapes:
                continue
            d1, d2 = normalize_shape(inv.param_shapes[param_id])
            param_dict[param_id] = _diffengine.make_parameter(
                d1, d2, offset, self.n_vars)
        return param_dict


# ---------------------------------------------------------------------------
# Atom converters: (expr, children, ctx) -> C expression
# ---------------------------------------------------------------------------

def _chain_add(children):
    """Chain multiple children with binary adds."""
    result = children[0]
    for child in children[1:]:
        result = _diffengine.make_add(result, child)
    return result


def _make_sparse_left_matmul(param_node, child, A):
    """Create sparse left matmul node, with optional param_node."""
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    args = [
        child,
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int32, copy=False),
        A.indptr.astype(np.int32, copy=False),
        A.shape[0],
        A.shape[1],
    ]
    if param_node is not None:
        args.insert(0, param_node)
    return _diffengine.make_sparse_left_matmul(*args)


def _make_dense_left_matmul(param_node, child, A):
    """Create dense left matmul node, with optional param_node."""
    m, n = normalize_shape(A.shape)
    args = [child, A.flatten(order='C'), m, n]
    if param_node is not None:
        args.insert(0, param_node)
    return _diffengine.make_dense_left_matmul(*args)


def _make_sparse_right_matmul(param_node, child, A):
    """Create sparse right matmul node, with optional param_node."""
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    args = [
        child,
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int32, copy=False),
        A.indptr.astype(np.int32, copy=False),
        A.shape[0],
        A.shape[1],
    ]
    if param_node is not None:
        args.insert(0, param_node)
    return _diffengine.make_sparse_right_matmul(*args)


def _make_dense_right_matmul(param_node, child, A):
    """Create dense right matmul node, with optional param_node."""
    m, n = normalize_shape(A.shape)
    args = [child, A.flatten(order='C'), m, n]
    if param_node is not None:
        args.insert(0, param_node)
    return _diffengine.make_dense_right_matmul(*args)


def _convert_matmul(expr, children, ctx):
    """Convert matrix multiplication A @ f(x), f(x) @ A, or X @ Y."""
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        A = left_arg.value
        # Recurse into the constant side to pick up any parameter nodes
        param_node = convert_expr(left_arg, ctx) if ctx.param_dict else None

        if sparse.issparse(A):
            return _make_sparse_left_matmul(param_node, children[1], A)
        else:
            return _make_dense_left_matmul(param_node, children[1], A)

    elif right_arg.is_constant():
        A = right_arg.value
        param_node = convert_expr(right_arg, ctx) if ctx.param_dict else None

        if sparse.issparse(A):
            return _make_sparse_right_matmul(param_node, children[0], A)
        else:
            return _make_dense_right_matmul(param_node, children[0], A)

    else:
        return _diffengine.make_matmul(children[0], children[1])


def _convert_multiply(expr, children, ctx):
    """Convert elementwise multiplication."""
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        if ctx.param_dict and left_arg.parameters():
            param_node = convert_expr(left_arg, ctx)
            if left_arg.size == 1:
                return _diffengine.make_param_scalar_mult(
                    param_node, children[1])
            return _diffengine.make_param_vector_mult(
                param_node, children[1])

        a = left_arg.value
        if sparse.issparse(a):
            a = a.todense()
        a = np.asarray(a, dtype=np.float64)
        if a.size == 1:
            scalar = float(a.flat[0])
            if scalar == 1.0:
                return children[1]
            return _diffengine.make_const_scalar_mult(children[1], scalar)
        return _diffengine.make_const_vector_mult(
            children[1], a.flatten(order='F'))

    elif right_arg.is_constant():
        if ctx.param_dict and right_arg.parameters():
            param_node = convert_expr(right_arg, ctx)
            if right_arg.size == 1:
                return _diffengine.make_param_scalar_mult(
                    param_node, children[0])
            return _diffengine.make_param_vector_mult(
                param_node, children[0])

        a = right_arg.value
        if sparse.issparse(a):
            a = a.todense()
        a = np.asarray(a, dtype=np.float64)
        if a.size == 1:
            scalar = float(a.flat[0])
            if scalar == 1.0:
                return children[0]
            return _diffengine.make_const_scalar_mult(children[0], scalar)
        return _diffengine.make_const_vector_mult(
            children[0], a.flatten(order='F'))

    return _diffengine.make_multiply(children[0], children[1])


def _convert_hstack(_expr, children, _ctx):
    return _diffengine.make_hstack(children)


def _extract_flat_indices_from_index(expr):
    """Extract flattened indices from CVXPY index expression."""
    parent_shape = expr.args[0].shape
    indices_per_dim = [np.arange(s.start, s.stop, s.step) for s in expr.key]

    if len(indices_per_dim) == 1:
        return indices_per_dim[0].astype(np.int32)
    elif len(indices_per_dim) == 2:
        return (
            np.add.outer(
                indices_per_dim[0], indices_per_dim[1] * parent_shape[0])
            .flatten(order="F")
            .astype(np.int32)
        )
    else:
        raise NotImplementedError("index with >2 dimensions not supported")


def _extract_flat_indices_from_special_index(expr):
    """Extract flattened indices from CVXPY special_index expression."""
    return np.reshape(
        expr._select_mat, expr._select_mat.size, order="F").astype(np.int32)


def _convert_rel_entr(expr, children, _ctx):
    """Convert rel_entr(x, y) = x * log(x/y) elementwise."""
    x_size = expr.args[0].size
    y_size = expr.args[1].size
    if x_size == y_size:
        return _diffengine.make_rel_entr(children[0], children[1])
    elif x_size > 1 and y_size == 1:
        return _diffengine.make_rel_entr_vector_scalar(
            children[0], children[1])
    elif x_size == 1 and y_size > 1:
        return _diffengine.make_rel_entr_scalar_vector(
            children[0], children[1])
    raise ValueError(
        f"rel_entr: incompatible sizes x={x_size}, y={y_size}")


def _convert_quad_form(expr, children, _ctx):
    """Convert quadratic form x.T @ P @ x."""
    P = expr.args[1]
    if not isinstance(P, cp.Constant):
        raise NotImplementedError(
            "quad_form requires P to be a constant matrix")
    P = P.value
    if not isinstance(P, sparse.csr_matrix):
        P = sparse.csr_matrix(P)
    return _diffengine.make_quad_form(
        children[0],
        P.data.astype(np.float64),
        P.indices.astype(np.int32),
        P.indptr.astype(np.int32),
        P.shape[0],
        P.shape[1],
    )


def _convert_reshape(expr, children, _ctx):
    if expr.order != "F":
        raise NotImplementedError(
            f"reshape with order='{expr.order}' not supported. "
            "Only order='F' (Fortran) is currently supported.")
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_reshape(children[0], d1, d2)


def _convert_broadcast(expr, children, _ctx):
    d1, d2 = expr.broadcast_shape
    d1_C, d2_C = _diffengine.get_expr_dimensions(children[0])
    if d1_C == d1 and d2_C == d2:
        return children[0]
    return _diffengine.make_broadcast(children[0], d1, d2)


def _convert_sum(expr, children, _ctx):
    axis = expr.axis if expr.axis is not None else -1
    return _diffengine.make_sum(children[0], axis)


def _convert_promote(expr, children, _ctx):
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_promote(children[0], d1, d2)


def _convert_index(expr, children, _ctx):
    idxs = _extract_flat_indices_from_index(expr)
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_index(children[0], d1, d2, idxs)


def _convert_special_index(expr, children, _ctx):
    idxs = _extract_flat_indices_from_special_index(expr)
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_index(children[0], d1, d2, idxs)


def _convert_prod(expr, children, _ctx):
    if expr.axis is None:
        return _diffengine.make_prod(children[0])
    elif expr.axis == 0:
        return _diffengine.make_prod_axis_zero(children[0])
    elif expr.axis == 1:
        return _diffengine.make_prod_axis_one(children[0])


def _convert_transpose(expr, children, _ctx):
    child_shape = normalize_shape(expr.args[0].shape)
    if 1 in child_shape:
        return _diffengine.make_reshape(
            children[0], child_shape[1], child_shape[0])
    return _diffengine.make_transpose(children[0])


def _convert_diag_vec(expr, children, _ctx):
    if expr.k != 0:
        raise NotImplementedError(
            "diag_vec with k != 0 not supported in diff engine")
    return _diffengine.make_diag_vec(children[0])


# ---------------------------------------------------------------------------
# Atom converter registry
# All converters have signature: (expr, children, ctx) -> C expression
# ---------------------------------------------------------------------------

ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda e, c, x: _diffengine.make_log(c[0]),
    "exp": lambda e, c, x: _diffengine.make_exp(c[0]),
    "NegExpression": lambda e, c, x: _diffengine.make_neg(c[0]),
    "Promote": _convert_promote,
    # N-ary
    "AddExpression": lambda e, c, x: _chain_add(c),
    # Reductions
    "Sum": _convert_sum,
    # Bivariate
    "multiply": _convert_multiply,
    "QuadForm": _convert_quad_form,
    "quad_over_lin": lambda e, c, x: _diffengine.make_quad_over_lin(c[0], c[1]),
    "rel_entr": _convert_rel_entr,
    # Matrix multiplication
    "MulExpression": _convert_matmul,
    # Power
    "Power": lambda e, c, x: _diffengine.make_power(c[0], float(e.p.value)),
    "PowerApprox": lambda e, c, x: _diffengine.make_power(c[0], float(e.p.value)),
    # Trigonometric
    "sin": lambda e, c, x: _diffengine.make_sin(c[0]),
    "cos": lambda e, c, x: _diffengine.make_cos(c[0]),
    "tan": lambda e, c, x: _diffengine.make_tan(c[0]),
    # Hyperbolic
    "sinh": lambda e, c, x: _diffengine.make_sinh(c[0]),
    "tanh": lambda e, c, x: _diffengine.make_tanh(c[0]),
    "asinh": lambda e, c, x: _diffengine.make_asinh(c[0]),
    "atanh": lambda e, c, x: _diffengine.make_atanh(c[0]),
    # Other elementwise
    "entr": lambda e, c, x: _diffengine.make_entr(c[0]),
    "logistic": lambda e, c, x: _diffengine.make_logistic(c[0]),
    "xexp": lambda e, c, x: _diffengine.make_xexp(c[0]),
    "normcdf": lambda e, c, x: _diffengine.make_normal_cdf(c[0]),
    # Indexing/slicing
    "index": _convert_index,
    "special_index": _convert_special_index,
    "reshape": _convert_reshape,
    "broadcast_to": _convert_broadcast,
    # Reductions
    "Prod": _convert_prod,
    "transpose": _convert_transpose,
    # Stack
    "Hstack": _convert_hstack,
    # Other matrix ops
    "Trace": lambda e, c, x: _diffengine.make_trace(c[0]),
    "diag_vec": _convert_diag_vec,
}


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def convert_expr(expr, ctx):
    """Convert a CVXPY expression to a C diff engine expression.

    Args:
        expr: CVXPY expression tree node
        ctx: ConvertContext with var_dict, param_dict, n_vars
    """
    # Variable lookup
    if isinstance(expr, cp.Variable):
        return ctx.var_dict[expr.id]

    # Parameter lookup
    if isinstance(expr, Parameter) and expr.id in ctx.param_dict:
        return ctx.param_dict[expr.id]

    # Constant (includes Parameters when param_dict is empty)
    if isinstance(expr, cp.Constant):
        c = expr.value
        if sparse.issparse(c):
            c = c.todense()
        c = np.asarray(c, dtype=np.float64)
        d1, d2 = normalize_shape(expr.shape)
        return _diffengine.make_constant(d1, d2, ctx.n_vars, c.flatten(order='F'))

    # Recursive case: atoms
    atom_name = type(expr).__name__
    if atom_name in ATOM_CONVERTERS:
        children = [convert_expr(arg, ctx) for arg in expr.args]
        C_expr = ATOM_CONVERTERS[atom_name](expr, children, ctx)

        # Dimension consistency check
        d1_C, d2_C = _diffengine.get_expr_dimensions(C_expr)
        d1_py, d2_py = normalize_shape(expr.shape)
        if d1_C != d1_py or d2_C != d2_py:
            raise ValueError(
                f"Dimension mismatch for '{atom_name}': "
                f"C ({d1_C}, {d2_C}) vs Python ({d1_py}, {d2_py})")

        return C_expr

    raise NotImplementedError(f"Atom '{atom_name}' not supported")

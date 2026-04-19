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

Adapter that converts a CVXPY Problem into a sparsediffpy.Problem by
translating the CVXPY expression tree into SparseDiffPy expressions.
"""
import numpy as np
import sparsediffpy as sp
from scipy import sparse
from sparsediffpy._core._constants import Constant as _SpConstant
from sparsediffpy._core._constants import SparseConstant as _SpSparseConstant

import cvxpy as cp
from cvxpy.reductions.inverse_data import InverseData


def _normalize_shape(shape):
    """Normalise a CVXPY shape to 2-D, prepending 1s (row convention).

    Matches CVXPY's broadcasting semantics: a 1-D shape `(N,)` behaves as a
    row `(1, N)` when broadcast against 2-D, and as a column in `A @ x`
    contexts — the latter is handled via a local reshape in MulExpression.
    """
    shape = tuple(shape)
    return (1,) * (2 - len(shape)) + shape


def _as_column(x):
    """Reshape a (1, n) row or (n, 1) column to a column (n, 1)."""
    if x.shape[1] == 1:
        return x
    return sp.reshape(x, x.shape[0] * x.shape[1], 1)


def _wrap_constant_value(value, shape):
    """Wrap a CVXPY constant value into a SparseDiffPy Constant/SparseConstant.

    Uses the CVXPY-declared `shape` (normalised to 2-D) so downstream operator
    dispatch sees a consistent shape for every constant node.
    """
    if sparse.issparse(value):
        return _SpSparseConstant(value)
    d1, d2 = _normalize_shape(shape)
    return _SpConstant(np.asarray(value, dtype=np.float64), (d1, d2))


def _convert_matmul(expr, children):
    # SparseDiffPy's `@` enforces `left.shape[1] == right.shape[0]` strictly;
    # the old C-level matmul was lenient and accepted any (1, n) / (n,) / (n, 1)
    # combination for vector-matmul, so no reshaping was needed there. The two
    # reshapes below are the minimal fixups to satisfy the strict check:
    #   1. RHS: CVXPY 1-D `(n,)` is stored as row (1, n); matmul needs a
    #      column (n, 1) on the right — covers both `A @ x` and `x @ y` dot.
    #   2. Result: `A @ x` with 1-D `x` yields (m, 1), but CVXPY declared the
    #      result 1-D which we normalise to a row (1, m).
    left, right = children
    if len(expr.args[1].shape) == 1 and right.shape[1] != 1:
        right = _as_column(right)
    result = left @ right
    if len(expr.shape) == 1 and result.shape[1] == 1 and result.shape[0] != 1:
        result = sp.reshape(result, 1, result.shape[0])
    return result


def _convert_transpose(expr, children):
    # For vectors ((1, n) or (n, 1)), transpose is a no-op in Fortran-order
    # flat storage, so use the cheap reshape; for true matrices, use the real
    # Transpose node which permutes elements.
    child_shape = _normalize_shape(expr.args[0].shape)
    if 1 in child_shape:
        return sp.reshape(children[0], child_shape[1], child_shape[0])
    return children[0].T


def _convert_reshape(expr, children):
    if expr.order != "F":
        raise NotImplementedError(
            f"reshape with order='{expr.order}' not supported. "
            "Only order='F' (Fortran) is currently supported."
        )
    d1, d2 = _normalize_shape(expr.shape)
    return sp.reshape(children[0], d1, d2)


def _convert_diag_vec(expr, children):
    if expr.k != 0:
        raise NotImplementedError(
            "diag_vec with k != 0 not supported in diff engine"
        )
    return sp.diag_vec(_as_column(children[0]))


def _convert_quad_form(expr, children):
    P = expr.args[1]
    if not isinstance(P, cp.Constant):
        raise NotImplementedError("quad_form requires P to be a constant matrix")
    P_val = P.value
    if not isinstance(P_val, sparse.csr_matrix):
        P_val = sparse.csr_matrix(P_val)
    return sp.quad_form(_as_column(children[0]), P_val)


def _convert_index(expr, children):
    parent_shape = expr.args[0].shape
    slices = [np.arange(s.start, s.stop, s.step) for s in expr.key]
    if len(slices) == 1:
        idxs = slices[0].astype(np.int32)
    elif len(slices) == 2:
        idxs = (
            np.add.outer(slices[0], slices[1] * parent_shape[0])
            .flatten(order="F")
            .astype(np.int32)
        )
    else:
        raise NotImplementedError("index with >2 dimensions not supported")
    return sp.index_flat(children[0], idxs, _normalize_shape(expr.shape))


def _convert_special_index(expr, children):
    idxs = np.reshape(
        expr._select_mat, expr._select_mat.size, order="F"
    ).astype(np.int32)
    return sp.index_flat(children[0], idxs, _normalize_shape(expr.shape))


def _sum_args(children):
    result = children[0]
    for c in children[1:]:
        result = result + c
    return result


def _elementwise(fn):
    return lambda expr, children: fn(children[0])


def _convert_promote(expr, children):
    return sp.broadcast(children[0], _normalize_shape(expr.shape))


def _convert_broadcast(expr, children):
    return sp.broadcast(children[0], tuple(expr.broadcast_shape))


_CONVERTERS = {
    # N-ary / unary
    "AddExpression": lambda expr, children: _sum_args(children),
    "NegExpression": lambda expr, children: -children[0],
    "multiply": lambda expr, children: children[0] * children[1],
    "MulExpression": _convert_matmul,

    # Structural / affine
    "Promote": _convert_promote,
    "broadcast_to": _convert_broadcast,
    "Sum": lambda expr, children: sp.sum(children[0], axis=expr.axis),
    "Prod": lambda expr, children: sp.prod(children[0], axis=expr.axis),
    "Power": lambda expr, children: sp.power(children[0], float(expr.p.value)),
    "PowerApprox": lambda expr, children: sp.power(children[0], float(expr.p.value)),
    "Trace": lambda expr, children: sp.trace(children[0]),
    "Hstack": lambda expr, children: sp.hstack(children),
    "transpose": _convert_transpose,
    "reshape": _convert_reshape,
    "diag_vec": _convert_diag_vec,
    "index": _convert_index,
    "special_index": _convert_special_index,

    # Bivariate
    "QuadForm": _convert_quad_form,
    "quad_over_lin": lambda expr, children: sp.quad_over_lin(children[0], children[1]),
    "rel_entr": lambda expr, children: sp.rel_entr(children[0], children[1]),

    # Elementwise unary
    "log": _elementwise(sp.log),
    "exp": _elementwise(sp.exp),
    "sin": _elementwise(sp.sin),
    "cos": _elementwise(sp.cos),
    "tan": _elementwise(sp.tan),
    "sinh": _elementwise(sp.sinh),
    "tanh": _elementwise(sp.tanh),
    "asinh": _elementwise(sp.asinh),
    "atanh": _elementwise(sp.atanh),
    "entr": _elementwise(sp.entr),
    "logistic": _elementwise(sp.logistic),
    "xexp": _elementwise(sp.xexp),
    "normcdf": _elementwise(sp.normal_cdf),
}


def _convert(expr, var_map, param_map):
    if isinstance(expr, cp.Variable):
        return var_map[expr.id]
    if isinstance(expr, cp.Parameter):
        return param_map[expr.id]
    if isinstance(expr, cp.Constant):
        return _wrap_constant_value(expr.value, expr.shape)

    atom_name = type(expr).__name__
    converter = _CONVERTERS.get(atom_name)
    if converter is None:
        raise NotImplementedError(f"Atom '{atom_name}' not supported")

    children = [_convert(arg, var_map, param_map) for arg in expr.args]
    result = converter(expr, children)

    target = _normalize_shape(expr.shape)
    if result.shape != target:
        raise ValueError(
            f"Dimension mismatch for atom '{atom_name}': "
            f"SparseDiff shape {result.shape} vs CVXPY shape {target}"
        )
    return result


def build_sparsediff_problem(
    cvxpy_problem: cp.Problem, verbose: bool = False
) -> sp.Problem:
    """Build a sparsediffpy.Problem from a CVXPY Problem.

    Variables are created in the order given by InverseData's id_map (sorted
    by offset) so the resulting flat-vector layout matches what Oracles sends
    to objective_forward / constraint_forward. Parameters are created in the
    order of cvxpy_problem.parameters() so Oracles.update_params' Fortran-flat
    concatenation aligns with the sparsediffpy.Problem's parameter layout.
    """
    inverse_data = InverseData(cvxpy_problem)
    scope = sp.Scope()

    var_map = {}
    for var_id, (_offset, _length) in sorted(
        inverse_data.id_map.items(), key=lambda kv: kv[1][0]
    ):
        d1, d2 = _normalize_shape(inverse_data.var_shapes[var_id])
        var_map[var_id] = scope.Variable(d1, d2)

    param_map = {}
    for cvxpy_param in cvxpy_problem.parameters():
        d1, d2 = _normalize_shape(inverse_data.param_shapes[cvxpy_param.id])
        sp_param = scope.Parameter(d1, d2)
        sp_param.value = np.asarray(cvxpy_param.value, dtype=np.float64)
        param_map[cvxpy_param.id] = sp_param

    obj_expr = _convert(cvxpy_problem.objective.expr, var_map, param_map)
    constraint_exprs = [
        _convert(c.expr, var_map, param_map) for c in cvxpy_problem.constraints
    ]

    return sp.Problem(obj_expr, constraint_exprs, verbose=verbose)

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

from typing import List, Optional, Tuple

import numpy as np
from numpy.exceptions import AxisError

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


def concatenate(arg_list, axis: Optional[int] = 0):
    assert axis is None or (isinstance(axis, int) and axis >= 0)
    return Concatenate(*(arg_list + [axis]))


class Concatenate(AffAtom):
    """Concatenate along an existing axis"""

    def __init__(self, *args) -> None:
        if isinstance(args[-1], int) or args[-1] is None:
            # Assume the last positional argument is axis
            axis = args[-1]
            args = args[:-1]
            self.axis = axis
        else:
            self.axis = None
        super().__init__(*args)

    def _supports_cpp(self) -> bool:
        return False

    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

    # Returns the concatenation of the values along the specified axis.
    def numeric(self, values):
        return np.concatenate(values, axis=self.axis)

    def get_data(self) -> List[Optional[int]]:
        return [self.axis]

    def validate_arguments(self) -> None:
        # Validates that the input shapes in `self.args` are suitable for
        # concatenation along a specified axis using numpy API with empty arrays
        self.shape_from_args()

    def shape_from_args(self) -> Tuple[int, ...]:
        try:
            return np.concatenate(
                [np.empty(arg.shape, dtype=np.dtype([])) for arg in self.args],
                axis=self.axis,
            ).shape
        except (ValueError, AxisError) as e:
            raise ValueError(f"Invalid arguments for cp.concatenate: {e}") from e

    def graph_implementation(
        self,
        arg_objs,
        shape: Tuple[int, ...],
        data=None,
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Concatenate the expressions along an existing axis.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom. In this case data wraps axis

        Returns
        -------
        tuple
            (LinOp for the objective, list of constraints)
        """
        return (lu.concatenate(arg_objs, shape, data[0]), [])

    def _verify_jacobian_args(self):
        return True

    def _jacobian(self):
        """Compute the Jacobian of concatenate."""

        result = {}
        output_shape = self.shape

        for idx, arg in enumerate(self.args):
            jac = arg.jacobian()

            if self.axis is None:
                # Simple case: just concatenate flat arrays
                flat_offset = sum(a.size for a in self.args[:idx])
                for k, (rows, cols, vals) in jac.items():
                    new_rows = rows + flat_offset
                    if k in result:
                        old_rows, old_cols, old_vals = result[k]
                        result[k] = (
                            np.concatenate([old_rows, new_rows]),
                            np.concatenate([old_cols, cols]),
                            np.concatenate([old_vals, vals]),
                        )
                    else:
                        result[k] = (new_rows, cols, vals)
            else:
                # General case: compute index mapping for this axis
                slices = [slice(None)] * len(output_shape)
                start = sum(a.shape[self.axis] for a in self.args[:idx])
                slices[self.axis] = slice(start, start + arg.shape[self.axis])

                output_indices = np.arange(self.size).reshape(output_shape, order='F')
                output_mapping = output_indices[tuple(slices)].ravel(order='F')

                for k, (rows, cols, vals) in jac.items():
                    new_rows = output_mapping[rows]
                    if k in result:
                        old_rows, old_cols, old_vals = result[k]
                        result[k] = (
                            np.concatenate([old_rows, new_rows]),
                            np.concatenate([old_cols, cols]),
                            np.concatenate([old_vals, vals]),
                        )
                    else:
                        result[k] = (new_rows, cols, vals)

        return result

    def _verify_hess_vec_args(self):
        return True

    def _hess_vec(self, vec):
        """Compute the Hessian-vector product for concatenate."""
        from scipy.sparse import coo_matrix

        result = {}
        output_shape = self.shape

        for idx, arg in enumerate(self.args):
            if self.axis is None:
                start = sum(a.size for a in self.args[:idx])
                arg_vec = vec[start:start + arg.size]
            else:
                slices = [slice(None)] * len(output_shape)
                start = sum(a.shape[self.axis] for a in self.args[:idx])
                slices[self.axis] = slice(start, start + arg.shape[self.axis])

                output_indices = np.arange(self.size).reshape(output_shape, order='F')
                output_mapping = output_indices[tuple(slices)].ravel(order='F')
                arg_vec = vec[output_mapping]

            arg_result = arg.hess_vec(arg_vec)
            for k, v in arg_result.items():
                if k in result:
                    old_rows, old_cols, old_vals = result[k]
                    new_rows, new_cols, new_vals = v
                    result[k] = (
                        np.concatenate([old_rows, new_rows]),
                        np.concatenate([old_cols, new_cols]),
                        np.concatenate([old_vals, new_vals]),
                    )
                else:
                    result[k] = v

        # Note: hess_vec may have duplicates if same variable pair appears
        # in multiple args, so we need to sum them
        for k in result:
            rows, cols, vals = result[k]
            var1, var2 = k
            hess = coo_matrix((vals, (rows, cols)), shape=(var1.size, var2.size))
            hess.sum_duplicates()
            result[k] = (hess.row, hess.col, hess.data)

        return result

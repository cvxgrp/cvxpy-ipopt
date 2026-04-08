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

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import lower_and_order_constraints
from cvxpy.reductions.matrix_stuffing import MatrixStuffing


class DiffengineMatrixStuffing(MatrixStuffing):
    """Construct matrices for cone problems using the diffengine (C autodiff) backend.

    Sibling of ConeMatrixStuffing that uses sparsediffpy's C autodiff engine
    to extract A, b, q, d, P matrices directly, bypassing the parametric
    tensor pipeline entirely.
    """
    CONSTRAINTS = 'ordered_constraints'

    def __init__(self, quad_obj: bool = False):
        self.quad_obj = quad_obj

    def accepts(self, problem):
        from cvxpy.reductions import cvx_attr2constr
        from cvxpy.reductions.utilities import are_args_affine

        valid_obj_curv = (self.quad_obj and problem.objective.expr.is_quadratic()) or \
            problem.objective.expr.is_affine()
        return (type(problem.objective) == Minimize
                and valid_obj_curv
                and not cvx_attr2constr.convex_attributes(problem.variables())
                and are_args_affine(problem.constraints)
                and problem.is_dpp())

    def apply(self, problem):
        from cvxpy.reductions.dcp2cone.diffengine_cone_program import (
            DiffengineConeProgram,
        )

        inverse_data = InverseData(problem)

        ordered_cons, cons_id_map = lower_and_order_constraints(problem.constraints)

        inverse_data.cons_id_map = cons_id_map
        inverse_data.constraints = ordered_cons
        inverse_data.minimize = type(problem.objective) == Minimize

        new_prob = DiffengineConeProgram.from_problem(
            problem, ordered_cons, inverse_data, self.quad_obj)

        return new_prob, inverse_data

    def invert(self, solution, inverse_data):
        """Retrieves a solution to the original problem."""
        var_map = inverse_data.var_offsets
        con_map = inverse_data.cons_id_map
        # Flip sign of opt val if maximize.
        opt_val = solution.opt_val
        if solution.status not in s.ERROR and not inverse_data.minimize:
            opt_val = -solution.opt_val

        primal_vars, dual_vars = {}, {}
        if solution.status not in s.SOLUTION_PRESENT:
            return Solution(solution.status, opt_val, primal_vars, dual_vars,
                            solution.attr)

        # Split vectorized variable into components.
        x_opt = list(solution.primal_vars.values())[0]
        for var_id, offset in var_map.items():
            shape = inverse_data.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            primal_vars[var_id] = np.reshape(x_opt[offset:offset+size], shape,
                                             order='F')

        # Remap dual variables if dual exists (problem is convex).
        if solution.dual_vars is not None:
            for old_con, new_con in con_map.items():
                con_obj = inverse_data.id2cons[old_con]
                shape = con_obj.shape
                # TODO rationalize Exponential.
                if shape == () or isinstance(con_obj, (ExpCone, SOC)):
                    dual_vars[old_con] = solution.dual_vars[new_con]
                else:
                    dual_vars[old_con] = np.reshape(
                        solution.dual_vars[new_con],
                        shape,
                        order='F'
                    )

        return Solution(solution.status, opt_val, primal_vars, dual_vars,
                        solution.attr)

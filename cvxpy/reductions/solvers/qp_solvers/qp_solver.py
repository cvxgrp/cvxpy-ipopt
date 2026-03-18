"""
Copyright 2017 Robin Verschueren

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

import time

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.error import SolverError
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solvers.solver import Solver


def _has_unsupported_cones(cone_dims) -> bool:
    """Check if cone dimensions contain cones unsupported by QP solvers."""
    return (cone_dims.exp > 0 or cone_dims.soc or cone_dims.psd
            or cone_dims.p3d or cone_dims.pnd)


def _get_unsupported_cone_message(cone_dims) -> str:
    """Get a descriptive message about unsupported cones."""
    unsupported = []
    if cone_dims.exp > 0:
        unsupported.append(f"{cone_dims.exp} exponential cones")
    if cone_dims.soc:
        unsupported.append(f"{len(cone_dims.soc)} second-order cones")
    if cone_dims.psd:
        unsupported.append(f"{len(cone_dims.psd)} PSD cones")
    if cone_dims.p3d:
        unsupported.append(f"{len(cone_dims.p3d)} 3D power cones")
    if cone_dims.pnd:
        unsupported.append(f"{len(cone_dims.pnd)} ND power cones")
    return ', '.join(unsupported)


class QpSolver(Solver):
    """
    A QP solver interface.

    QP solvers accept ParamConeProg with only Zero and NonNeg constraints
    (i.e., equality and inequality constraints) and convert them to the
    standard QP form:

        minimize      1/2 x' P x + q' x
        subject to    A x =  b
                      F x <= g
    """
    # Every QP solver supports Zero and NonNeg constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    IS_MIP = "IS_MIP"

    def supports_quad_obj(self) -> bool:
        """QP solvers support quadratic objectives."""
        return True

    def accepts(self, problem):
        return (isinstance(problem, ParamConeProg)
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes([problem.x])
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in problem.constraints)
                and not _has_unsupported_cones(problem.cone_dims))

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.

        Converts a ParamConeProg (with only Zero and NonNeg constraints) to QP form:

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g
        """
        if not self.accepts(problem):
            if _has_unsupported_cones(problem.cone_dims):
                raise SolverError(
                    f"QP solver {self.name()} cannot handle: "
                    f"{_get_unsupported_cone_message(problem.cone_dims)}. "
                    "This may indicate a bug in solver selection. Please report this issue."
                )

        t_total = time.perf_counter()

        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        data[QpSolver.DIMS] = problem.cone_dims
        inv_data[QpSolver.DIMS] = data[QpSolver.DIMS]
        inv_data[QpSolver.IS_MIP] = problem.is_mixed_integer()
        data[s.PARAM_PROB] = problem

        # Apply parameters with quadratic objective
        t0 = time.perf_counter()
        P, q, d, AF, bg = problem.apply_parameters(quad_obj=True)
        t1 = time.perf_counter()
        s.LOGGER.info('[QpSolver.apply] apply_parameters: %.4f s', t1 - t0)
        inv_data[s.OFFSET] = d

        # Get number of variables
        n = problem.x.size
        len_eq = problem.cone_dims.zero
        len_leq = problem.cone_dims.nonneg

        # Store constraint lists for dual variable mapping
        # Constraints are ordered: Zero (equality) first, then NonNeg (inequality)
        eq_constrs = [c for c in problem.constraints if type(c) == Zero]
        ineq_constrs = [c for c in problem.constraints if type(c) == NonNeg]
        inv_data[self.EQ_CONSTR] = eq_constrs
        inv_data[self.NEQ_CONSTR] = ineq_constrs

        # Store combined constraint matrix for solvers that vstack A and F
        # (OSQP, HiGHS, QPALM). They can build their combined matrix directly
        # from AF without the split+negate+vstack overhead.
        t0 = time.perf_counter()
        data['AF'] = AF
        data['bg'] = bg
        data['len_eq'] = len_eq

        # Split into equality and inequality constraints
        # (needed by solvers that use A/F separately: GUROBI, CPLEX, etc.)
        if len_eq > 0:
            A = AF[:len_eq, :]
            b = -bg[:len_eq]
        else:
            A, b = sp.csr_array((0, n)), -np.array([])

        if len_leq > 0:
            F = -AF[len_eq:, :]
            g = bg[len_eq:]
        else:
            F, g = sp.csr_array((0, n)), -np.array([])

        # Create dictionary with problem data
        data[s.P] = P
        data[s.Q] = q
        data[s.A] = A
        data[s.B] = b
        data[s.F] = F
        data[s.G] = g
        t1 = time.perf_counter()
        s.LOGGER.info('[QpSolver.apply] constraint splitting + data dict: %.4f s', t1 - t0)

        data[s.BOOL_IDX] = [t[0] for t in problem.x.boolean_idx]
        data[s.INT_IDX] = [t[0] for t in problem.x.integer_idx]
        data[s.LOWER_BOUNDS] = problem.lower_bounds
        data[s.UPPER_BOUNDS] = problem.upper_bounds
        data['n_var'] = n
        data['n_eq'] = A.shape[0]
        data['n_ineq'] = F.shape[0]

        s.LOGGER.info('[QpSolver.apply] total: %.4f s', time.perf_counter() - t_total)
        return data, inv_data

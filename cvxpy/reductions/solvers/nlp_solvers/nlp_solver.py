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
"""

from time import time

import numpy as np

from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)


class NLPsolver(Solver):
    """
    A non-linear programming (NLP) solver.
    """
    REQUIRES_CONSTR = False
    MIP_CAPABLE = False

    def accepts(self, problem):
        """
        Only accepts disciplined nonlinear programs.
        """
        return problem.is_dnlp()

    def apply(self, problem):
        """
        Construct NLP problem data stored in a dictionary.
        The NLP has the following form

            minimize      f(x)
            subject to    g^l <= g(x) <= g^u
                          x^l <= x <= x^u
        where f and g are non-linear (and possibly non-convex) functions
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        return data, inv_data

    def _prepare_data_and_inv_data(self, problem):
        data = dict()
        bounds = Bounds(problem)
        inverse_data = InverseData(bounds.new_problem)
        inverse_data.offset = 0.0
        data["problem"] = bounds.new_problem
        data["cl"], data["cu"] = bounds.cl, bounds.cu
        data["lb"], data["ub"] = bounds.lb, bounds.ub
        data["x0"] = bounds.x0
        oracles = Oracles(bounds.new_problem, bounds.x0, len(bounds.cl))
        data["objective"] = oracles.objective
        data["gradient"] = oracles.gradient
        data["constraints"] = oracles.constraints
        data["jacobian"] = oracles.jacobian
        data["jacobianstructure"] = oracles.jacobianstructure
        data["hessian"] = oracles.hessian
        data["hessianstructure"] = oracles.hessianstructure
        data["oracles"] = oracles
        return problem, data, inverse_data

class Bounds():
    def __init__(self, problem):
        self.problem = problem
        self.main_var = problem.variables()
        self.get_constraint_bounds()
        self.get_variable_bounds()
        self.construct_initial_point()

    def get_constraint_bounds(self):
        """
        Get constraint bounds for all constraints.
        Also converts inequalities to nonneg form,
        as well as equalities to zero constraints and forms
        a new problem from the canonicalized problem.
        """
        lower, upper = [], []
        new_constr = []
        for constraint in self.problem.constraints:
            if isinstance(constraint, Equality):
                lower.extend([0.0] * constraint.size)
                upper.extend([0.0] * constraint.size)
                new_constr.append(lower_equality(constraint))
            elif isinstance(constraint, Inequality):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(lower_ineq_to_nonneg(constraint))
            elif isinstance(constraint, NonPos):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(nonpos2nonneg(constraint))
        canonicalized_prob = self.problem.copy([self.problem.objective, new_constr])
        self.new_problem = canonicalized_prob
        self.cl = np.array(lower)
        self.cu = np.array(upper)

    def get_variable_bounds(self):
        """
        Get variable bounds for all variables.
        Also takes into account nonneg/nonpos attributes.
        """
        var_lower, var_upper = [], []
        for var in self.main_var:
            size = var.size
            if var.bounds:
                lb = var.bounds[0].flatten(order='F')
                ub = var.bounds[1].flatten(order='F')
                if var.is_nonneg():
                    lb = np.maximum(lb, 0)
                if var.is_nonpos():
                    ub = np.minimum(ub, 0)
                var_lower.extend(lb)
                var_upper.extend(ub)
            else:
                # No bounds specified, use infinite bounds or bounds
                # set by the nonnegative or nonpositive attribute
                if var.is_nonneg():
                    var_lower.extend([0.0] * size)
                else:
                    var_lower.extend([-np.inf] * size)
                if var.is_nonpos():
                    var_upper.extend([0.0] * size)
                else:
                    var_upper.extend([np.inf] * size)
        self.lb = np.array(var_lower)
        self.ub = np.array(var_upper)
    

    def construct_initial_point(self):
        """ Loop through all variables and collect the intial point."""
        x0 = []
        for var in self.main_var:
            if var.value is None:
                raise ValueError("Variable %s has no value. This is a bug and should be reported."
                                  % var.name())

            x0.append(np.atleast_1d(var.value).flatten(order='F'))
        self.x0 = np.concatenate(x0, axis=0)

class Oracles():
    """
    Oracle interface for NLP solvers using the C-based diff engine.

    Provides function and derivative oracles (objective, gradient, constraints,
    jacobian, hessian) by wrapping the C_problem class from dnlp_diff_engine.
    """

    def __init__(self, problem, initial_point, num_constraints):
        # Lazy import to avoid circular dependency at module load time
        from dnlp_diff_engine import C_problem

        self.c_problem = C_problem(problem)
        start = time()
        self.c_problem.init_derivatives()
        self.time_init_derivatives = time() - start
        self.initial_point = initial_point
        self.num_constraints = num_constraints
        self.iterations = 0

        # Cached sparsity structures
        self._jac_structure = None
        self._hess_structure = None

        self.time_jacobian = 0.0
        self.time_jacobian_c = 0.0
        self.time_hessian_c = 0.0
     
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return self.c_problem.objective_forward(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        self.c_problem.objective_forward(x)
        return self.c_problem.gradient()

    def constraints(self, x):
        """Returns the constraint values."""
        return self.c_problem.constraint_forward(x)

    def jacobian(self, x):
        """Returns the Jacobian values in COO format at the sparsity structure."""
        self.c_problem.constraint_forward(x)

        start = time()
        jac_csr = self.c_problem.jacobian()
        self.time_jacobian_c += time() - start
        jac_coo = jac_csr.tocoo()

        if self._jac_structure is None:
            # First call - return values directly
            return jac_coo.data

        # Extract values at the known sparsity pattern
        rows_struct, cols_struct = self._jac_structure
        jac_dense = jac_csr.toarray()
        return np.array([jac_dense[r, c] for r, c in zip(rows_struct, cols_struct)])

    def jacobianstructure(self):
        """Returns the sparsity structure of the Jacobian."""
        if self._jac_structure is not None:
            return self._jac_structure

        # Evaluate at initial point to get structure
        self.c_problem.constraint_forward(self.initial_point)
        jac_csr = self.c_problem.jacobian()
        jac_coo = jac_csr.tocoo()

        self._jac_structure = (
            jac_coo.row.astype(np.int32),
            jac_coo.col.astype(np.int32)
        )
        return self._jac_structure

    def hessian(self, x, duals, obj_factor):
        """Returns the lower triangular Hessian values in COO format."""
        self.c_problem.objective_forward(x)
        if self.num_constraints > 0:
            self.c_problem.constraint_forward(x)
        start = time()
        hess_csr = self.c_problem.hessian(obj_factor, duals)
        self.time_hessian_c += time() - start
        hess_coo = hess_csr.tocoo()

        if self._hess_structure is None:
            # First call - extract lower triangular and return
            mask = hess_coo.row >= hess_coo.col
            return hess_coo.data[mask]

        # Extract values at the known sparsity pattern
        rows_struct, cols_struct = self._hess_structure
        hess_dense = hess_csr.toarray()
        return np.array([hess_dense[r, c] for r, c in zip(rows_struct, cols_struct)])

    def hessianstructure(self):
        """Returns the sparsity structure of the lower triangular Hessian."""
        if self._hess_structure is not None:
            return self._hess_structure

        # Evaluate at initial point with unit vectors to get structure
        self.c_problem.objective_forward(self.initial_point)
        if self.num_constraints > 0:
            self.c_problem.constraint_forward(self.initial_point)
            duals = np.ones(self.num_constraints)
        else:
            duals = np.array([])
        hess_csr = self.c_problem.hessian(1.0, duals)
        hess_coo = hess_csr.tocoo()

        # Keep only lower triangular
        mask = hess_coo.row >= hess_coo.col
        self._hess_structure = (
            hess_coo.row[mask].astype(np.int32),
            hess_coo.col[mask].astype(np.int32)
        )
        return self._hess_structure

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""
        self.iterations = iter_count

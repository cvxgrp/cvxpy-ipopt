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
import numpy as np

from cvxpy import error
from cvxpy import settings as s
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.nlp_solvers.copt_nlpif import COPT as COPT_nlp
from cvxpy.reductions.solvers.nlp_solvers.ipopt_nlpif import IPOPT as IPOPT_nlp
from cvxpy.reductions.solvers.nlp_solvers.knitro_nlpif import KNITRO as KNITRO_nlp
from cvxpy.reductions.solvers.nlp_solvers.uno_nlpif import UNO as UNO_nlp
from cvxpy.reductions.solvers.solving_chain import SolvingChain


def _build_nlp_chain(problem, solver, kwargs):
    """Build the NLP reduction chain and return (SolvingChain, kwargs).

    Solver selection may mutate kwargs (e.g., Knitro algorithm, Uno preset).
    """
    if type(problem.objective) == Maximize:
        reductions = [FlipObjective()]
    else:
        reductions = []
    reductions = reductions + [CvxAttr2Constr(reduce_bounds=False), Dnlp2Smooth()]

    if solver is s.IPOPT or solver is None:
        reductions.append(IPOPT_nlp())
    elif "knitro" in solver.lower():
        if solver == "knitro_ipm":
            kwargs["algorithm"] = 1
        elif solver == "knitro_sqp":
            kwargs["algorithm"] = 4
        elif solver == "knitro_alm":
            kwargs["algorithm"] = 6
        reductions.append(KNITRO_nlp())
    elif solver is s.COPT:
        reductions.append(COPT_nlp())
    elif "uno" in solver.lower():
        if solver.lower() == "uno_ipm":
            kwargs["preset"] = "ipopt"
            kwargs["linear_solver"] = "MUMPS"
        elif solver.lower() == "uno_sqp":
            kwargs["preset"] = "filtersqp"
        reductions.append(UNO_nlp())
    else:
        raise error.SolverError(
            "Solver %s is not supported for NLP problems." % solver
        )

    return SolvingChain(reductions=reductions), kwargs


def _set_nlp_initial_point(problem):
    """Construct an initial point for variables without a user-specified value.

    Uses get_bounds() which incorporates sign attributes (nonneg, nonpos, etc.).
    If both lb and ub are finite, initialize to their midpoint. If only one is
    finite, initialize one unit from the bound. Otherwise, initialize to zero.
    """
    for var in problem.variables():
        if var.value is not None:
            continue

        lb, ub = var.get_bounds()

        lb_finite = np.isfinite(lb)
        ub_finite = np.isfinite(ub)
        # Replace infs with zero for arithmetic
        lb0 = np.where(lb_finite, lb, 0.0)
        ub0 = np.where(ub_finite, ub, 0.0)
        # Midpoint if both finite, one from bound if only one finite, zero if none
        init = (lb_finite * ub_finite * 0.5 * (lb0 + ub0) +
                lb_finite * (~ub_finite) * (lb0 + 1.0) +
                (~lb_finite) * ub_finite * (ub0 - 1.0))
        # Broadcast to variable shape (handles scalar bounds)
        init = np.broadcast_to(init, var.shape).copy()
        var.save_value(init)


def _set_random_nlp_initial_point(problem, run, user_initials):
    """Generate a random initial point for DNLP problems.

    A variable is initialized randomly if:
    1. 'sample_bounds' is set for that variable.
    2. The initial value specified by the user is None, 'sample_bounds' is not
       set, but the variable has both finite lower and upper bounds.

    Parameters
    ----------
    problem : Problem
    run : int
        Current run index (0-based).
    user_initials : dict
        On run 0, will be populated with user-specified initial values.
        On subsequent runs, used to restore user-specified values.
    """
    # Store user-specified initial values on the first run
    if run == 0:
        user_initials.clear()
        for var in problem.variables():
            if var.sample_bounds is not None:
                user_initials[var.id] = None
            else:
                user_initials[var.id] = var.value

    for var in problem.variables():
        # Skip variables with user-specified initial value
        # (note that any variable with sample bounds set will have
        #  user_initials[var.id] == None)
        if user_initials[var.id] is not None:
            # Reset to user-specified initial value from last solve
            var.value = user_initials[var.id]
            continue
        else:
            # Reset to None from last solve
            var.value = None

        # Set sample_bounds to variable bounds if sample_bounds is None
        # and variable bounds (possibly infinite) are set
        if var.sample_bounds is None and var.bounds is not None:
            var.sample_bounds = var.bounds

        # Sample initial value if sample_bounds is set
        if var.sample_bounds is not None:
            low, high = var.sample_bounds
            if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                raise ValueError(
                    "Variable %s has non-finite sample_bounds %s. Cannot generate"
                    " random initial point. Either add sample bounds or set the value. "
                    % (var.name(), var.sample_bounds)
                )

            initial_val = np.random.uniform(low=low, high=high, size=var.shape)
            var.save_value(initial_val)


def solve_nlp(problem, solver, warm_start, verbose, **kwargs):
    """Solve an NLP problem using the DNLP reduction chain.

    Parameters
    ----------
    problem : Problem
        A DNLP-valid problem.
    solver : str or None
        Solver name (e.g., 'IPOPT', 'knitro_sqp').
    warm_start : bool
        Whether to warm-start the solver.
    verbose : bool
        Whether to print solver output.
    **kwargs
        Additional solver options, including 'best_of'.

    Returns
    -------
    float
        The optimal problem value.
    """
    nlp_chain, kwargs = _build_nlp_chain(problem, solver, kwargs)
    best_of = kwargs.pop("best_of", 1)

    # Standard single solve
    if best_of == 1:
        _set_nlp_initial_point(problem)
        canon_problem, inverse_data = nlp_chain.apply(problem=problem)
        solution = nlp_chain.solver.solve_via_data(canon_problem, warm_start,
                                                   verbose, solver_opts=kwargs)
        problem.unpack_results(solution, nlp_chain, inverse_data)
        return problem.value

    # Best-of-N solve
    if (not isinstance(best_of, int)) or best_of < 1:
        raise ValueError("best_of must be a positive integer.")

    best_obj, best_solution = float("inf"), None
    all_objs = np.zeros(shape=(best_of,))
    user_initials = {}

    for run in range(best_of):
        print("Starting NLP solve %d of %d" % (run + 1, best_of))
        _set_random_nlp_initial_point(problem, run, user_initials)
        canon_problem, inverse_data = nlp_chain.apply(problem=problem)
        solution = nlp_chain.solver.solve_via_data(canon_problem, warm_start,
                                                   verbose, solver_opts=kwargs)

        # Unpack to get the objective value in the original problem space
        problem.unpack_results(solution, nlp_chain, inverse_data)
        obj_value = problem.objective.value

        all_objs[run] = obj_value
        if obj_value < best_obj:
            best_obj = obj_value
            print("best_obj: ", best_obj)
            best_solution = solution

    # Unpack best solution
    if type(problem.objective) == Maximize:
        all_objs = -all_objs

    # Propagate all objective values to the user
    best_solution['all_objs_from_best_of'] = all_objs
    problem.unpack_results(best_solution, nlp_chain, inverse_data)
    return problem.value

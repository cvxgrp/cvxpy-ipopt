import cyipopt
import numpy as np
from reduction_classes import HS071

import cvxpy as cp


# Example usage for HS071 problem setup
def create_hs071_problem():
    """Creates the classic HS071 optimization problem."""
    # Variables
    x = cp.Variable(4)
    
    # Objective: minimize x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]
    objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])
    
    # Constraints
    constraints = [
        x[0]*x[1]*x[2]*x[3] >= 25,  # Product constraint
        cp.sum_squares(x) == 40,    # Sum of squares constraint
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    return problem

lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]

cl = [0, 0]
cu = [np.inf, 0]

x0 = [1.0, 5.0, 5.0, 1.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(cl),
   problem_obj=HS071(create_hs071_problem()),
   lb=lb,
   ub=ub,
   cl=cl,
   cu=cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")

x, info = nlp.solve(x0)
print(x)
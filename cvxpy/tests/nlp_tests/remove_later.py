import cvxpy as cp
import numpy as np

x = cp.Variable(1)
y = cp.Variable(1, bounds=[0, np.inf])
z = cp.Variable(1, bounds=[0, np.inf])

objective = cp.Maximize(x)

constraints = [
    x + y + z == 1,
    x**2 + y**2 - z**2 <= 0,
    x**2 - cp.multiply(y, z) <= 0
]
problem = cp.Problem(objective, constraints)
problem.solve(solver="UNO", nlp=True)
assert problem.status == cp.OPTIMAL
assert np.allclose(x.value, np.array([0.32699284]))
assert np.allclose(y.value, np.array([0.25706586]))
assert np.allclose(z.value, np.array([0.4159413]))
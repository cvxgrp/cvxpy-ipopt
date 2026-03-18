# formulate a least squares problem and compare the backends time.
import logging
import time

import numpy as np

import cvxpy as cp
import cvxpy.settings as s

# Enable logging so all s.LOGGER.info() calls are visible
s.LOGGER.setLevel(logging.INFO)
s.LOGGER.addHandler(logging.StreamHandler())

np.random.seed(0)
m, n = 20000, 4000
A = np.random.randn(m, n)
b = np.random.randn(m)

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

start_time = time.time()
prob.get_problem_data(cp.OSQP, canon_backend='DIFFENGINE', verbose=True)
diffengine_time = time.time() - start_time

print(f"\nDIFFENGINE total time: {diffengine_time:.4f} seconds")

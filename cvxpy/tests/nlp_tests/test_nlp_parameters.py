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
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestNlpParameters:

    def test_parameter_least_squares(self):
        """min ||A @ x - b||^2, x nonneg with parametric A and b."""
        m, n = 50, 10
        np.random.seed(0)
        A1 = np.random.rand(m, n)
        b1 = np.random.rand(m)
        A2 = np.random.rand(m, n)
        b2 = np.random.rand(m)

        # Solve with hardcoded values
        x = cp.Variable(n, nonneg=True)
        prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A1 @ x - b1)))
        prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A2 @ x - b2)))
        x.value = None
        prob1.solve(nlp=True, solver='IPOPT')
        hardcoded_sol1 = x.value
        x.value = None
        prob2.solve(nlp=True, solver='IPOPT')
        hardcoded_sol2 = x.value

        # Solve with parameters
        A = cp.Parameter((m, n), value=A1)
        b = cp.Parameter(m, value=b1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol1 = x.value
        A.value = A2
        b.value = b2
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol2 = x.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0

    def test_parameter_entropy_maximization(self):
        """max sum(entr(x)) s.t. A @ x <= b, sum(x) == 1, x >= 0."""
        m, n = 10, 5
        np.random.seed(0)
        A1 = np.abs(np.random.rand(m, n))
        b1 = np.ones(m)
        A2 = np.abs(np.random.rand(m, n))
        b2 = np.ones(m) * 0.8

        # Solve with hardcoded values
        x = cp.Variable(n, nonneg=True)
        constraints1 = [A1 @ x <= b1, cp.sum(x) == 1]
        constraints2 = [A2 @ x <= b2, cp.sum(x) == 1]
        prob1 = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), constraints1)
        prob2 = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), constraints2)
        x.value = None
        prob1.solve(nlp=True, solver='IPOPT')
        hardcoded_sol1 = x.value
        x.value = None
        prob2.solve(nlp=True, solver='IPOPT')
        hardcoded_sol2 = x.value

        # Solve with parameters
        A = cp.Parameter((m, n), value=A1)
        b = cp.Parameter(m, value=b1)
        constraints = [A @ x <= b, cp.sum(x) == 1]
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), constraints)
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol1 = x.value
        A.value = A2
        b.value = b2
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol2 = x.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0

    def test_parameter_log_sum_exp(self):
        """min log_sum_exp(A @ x + b) s.t. -1 <= x <= 1."""
        m, n = 10, 5
        np.random.seed(0)
        A1 = np.random.randn(m, n)
        b1 = np.random.randn(m)
        A2 = np.random.randn(m, n)
        b2 = np.random.randn(m)

        # Solve with hardcoded values
        x = cp.Variable(n, bounds=[-1, 1])
        prob1 = cp.Problem(cp.Minimize(cp.log_sum_exp(A1 @ x + b1)))
        prob2 = cp.Problem(cp.Minimize(cp.log_sum_exp(A2 @ x + b2)))
        x.value = None
        prob1.solve(nlp=True, solver='IPOPT')
        hardcoded_sol1 = x.value
        x.value = None
        prob2.solve(nlp=True, solver='IPOPT')
        hardcoded_sol2 = x.value

        # Solve with parameters
        A = cp.Parameter((m, n), value=A1)
        b = cp.Parameter(m, value=b1)
        prob = cp.Problem(cp.Minimize(cp.log_sum_exp(A @ x + b)))
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol1 = x.value
        A.value = A2
        b.value = b2
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol2 = x.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0

    def test_parameter_right_matmul(self):
        """min ||X @ A - B||_F^2, X nonnegative with parametric A and B."""
        m, n, p = 5, 5, 20
        np.random.seed(0)
        A1 = np.random.rand(n, p)
        B1 = np.random.rand(m, p)
        A2 = np.random.rand(n, p)
        B2 = np.random.rand(m, p)

        # Solve with hardcoded values
        X = cp.Variable((m, n), nonneg=True)
        prob1 = cp.Problem(cp.Minimize(cp.sum_squares(X @ A1 - B1)))
        prob2 = cp.Problem(cp.Minimize(cp.sum_squares(X @ A2 - B2)))
        X.value = None
        prob1.solve(nlp=True, solver='IPOPT')
        hardcoded_sol1 = X.value
        X.value = None
        prob2.solve(nlp=True, solver='IPOPT')
        hardcoded_sol2 = X.value

        # Solve with parameters
        A = cp.Parameter((n, p), value=A1)
        B = cp.Parameter((m, p), value=B1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X @ A - B)))
        X.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol1 = X.value
        A.value = A2
        B.value = B2
        X.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol2 = X.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0

    def test_parameter_shared_across_expressions(self):
        """min ||A @ x - b||^2 s.t. sum(A @ x) == 1, x nonneg. A in obj and constraint."""
        m, n = 20, 5
        np.random.seed(0)
        A1 = np.random.rand(m, n)
        b1 = np.random.rand(m)
        A2 = np.random.rand(m, n)
        b2 = np.random.rand(m)

        # Solve with hardcoded values
        x = cp.Variable(n, nonneg=True)
        prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A1 @ x - b1)),
                           [cp.sum(A1 @ x) == 1])
        prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A2 @ x - b2)),
                           [cp.sum(A2 @ x) == 1])
        x.value = None
        prob1.solve(nlp=True, solver='IPOPT')
        hardcoded_sol1 = x.value
        x.value = None
        prob2.solve(nlp=True, solver='IPOPT')
        hardcoded_sol2 = x.value

        # Solve with parameters
        A = cp.Parameter((m, n), value=A1)
        b = cp.Parameter(m, value=b1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)),
                          [cp.sum(A @ x) == 1])
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol1 = x.value
        A.value = A2
        b.value = b2
        x.value = None
        prob.solve(nlp=True, solver='IPOPT')
        param_sol2 = x.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0


    # should have tests for sum(a * log(X)) with a scalar parameter,
    # vector parameter, matrix parameter.
    # also a * sum(log(X)). Should also have tests when we multiply from
    # the right instead of the left.
    # should have a test where we explicitly call cp.broadcast(parameter, shape)
    # and cp.promote(parameter, shape)
    # should also add factor model fitting test (raises an error right now I think)

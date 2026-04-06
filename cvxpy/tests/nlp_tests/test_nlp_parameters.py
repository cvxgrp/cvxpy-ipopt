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
class test_nlp_parameters:

    def test_parameter_least_squares(self):
        """min ||A @ x - b||^2 with parametric A and b, compared to Clarabel."""
        m, n = 50, 10
        np.random.seed(0)
        A = cp.Parameter((m, n), value=np.random.rand(m, n))
        x = cp.Variable(n)
        b = cp.Parameter(m, value=np.random.rand(m))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

        prob.solve(nlp=True, solver='IPOPT')
        nlp_sol = x.value.copy()
        prob.solve(solver='CLARABEL')
        assert np.allclose(nlp_sol, x.value, atol=1e-5)

        A.value = np.random.rand(m, n)
        b.value = np.random.rand(m)
        prob.solve(nlp=True, solver='IPOPT')
        nlp_sol = x.value.copy()
        prob.solve(solver='CLARABEL')
        assert np.allclose(nlp_sol, x.value, atol=1e-5)

    def test_parameter_entropy_maximization(self):
        """max sum(entr(x)) s.t. A @ x <= b, sum(x) == 1, x >= 0."""
        m, n = 10, 5
        np.random.seed(42)
        A = cp.Parameter((m, n), value=np.abs(np.random.rand(m, n)))
        b = cp.Parameter(m, value=np.ones(m))
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Maximize(cp.sum(cp.entr(x))),
            [A @ x <= b, cp.sum(x) == 1, x >= 0],
        )

        prob.solve(nlp=True, solver='IPOPT')
        nlp_val = prob.value
        nlp_sol = x.value.copy()
        prob.solve(solver='CLARABEL')
        assert np.isclose(nlp_val, prob.value, atol=1e-5)
        assert np.allclose(nlp_sol, x.value, atol=1e-5)

        A.value = np.abs(np.random.rand(m, n))
        b.value = np.ones(m) * 0.8
        prob.solve(nlp=True, solver='IPOPT')
        nlp_val = prob.value
        nlp_sol = x.value.copy()
        prob.solve(solver='CLARABEL')
        assert np.isclose(nlp_val, prob.value, atol=1e-5)
        assert np.allclose(nlp_sol, x.value, atol=1e-5)

    def test_parameter_log_sum_exp(self):
        """min log_sum_exp(A @ x + b) s.t. -1 <= x <= 1."""
        m, n = 10, 5
        np.random.seed(7)
        A = cp.Parameter((m, n), value=np.random.randn(m, n))
        b = cp.Parameter(m, value=np.random.randn(m))
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.log_sum_exp(A @ x + b)),
            [x >= -1, x <= 1],
        )

        prob.solve(nlp=True, solver='IPOPT')
        nlp_val = prob.value
        nlp_sol = x.value.copy()
        prob.solve(solver='CLARABEL')
        assert np.isclose(nlp_val, prob.value, atol=1e-5)
        assert np.allclose(nlp_sol, x.value, atol=1e-4)

        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        prob.solve(nlp=True, solver='IPOPT')
        nlp_val = prob.value
        nlp_sol = x.value.copy()
        prob.solve(solver='CLARABEL')
        assert np.isclose(nlp_val, prob.value, atol=1e-5)
        assert np.allclose(nlp_sol, x.value, atol=1e-4)

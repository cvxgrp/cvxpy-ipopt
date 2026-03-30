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
class Test_NLP_parameters:

    def test_scalar_parameter(self):
        """min p * x^2 + x, analytical solution: val = -1/(4p)."""
        x = cp.Variable()
        p = cp.Parameter(value=2.0)
        prob = cp.Problem(cp.Minimize(p * x**2 + x), [x >= -5])

        prob.solve(nlp=True, solver='IPOPT')
        assert np.isclose(prob.value, -1.0 / (4 * 2.0), atol=1e-4)

        p.value = 4.0
        prob.solve(nlp=True, solver='IPOPT')
        assert np.isclose(prob.value, -1.0 / (4 * 4.0), atol=1e-4)

    def test_vector_parameter(self):
        """min p @ x with simplex constraint."""
        x = cp.Variable(2)
        p = cp.Parameter(2, value=[1.0, 2.0])
        prob = cp.Problem(cp.Minimize(p @ x), [x >= 0, cp.sum(x) == 1])

        prob.solve(nlp=True, solver='IPOPT')
        assert np.isclose(prob.value, 1.0, atol=1e-4)
        assert np.allclose(x.value, [1.0, 0.0], atol=1e-3)

        p.value = [3.0, 1.0]
        prob.solve(nlp=True, solver='IPOPT')
        assert np.isclose(prob.value, 1.0, atol=1e-4)
        assert np.allclose(x.value, [0.0, 1.0], atol=1e-3)

    def test_matrix_parameter(self):
        """min ||A @ x - b||^2 with parametric A."""
        A = cp.Parameter((2, 2), value=np.eye(2))
        x = cp.Variable(2)
        b = np.array([1.0, 2.0])
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(A @ x - b)),
            [x >= -10, x <= 10])

        prob.solve(nlp=True, solver='IPOPT')
        assert np.allclose(x.value, [1.0, 2.0], atol=1e-3)

        A.value = np.array([[0.0, 1.0], [1.0, 0.0]])
        prob.solve(nlp=True, solver='IPOPT')
        assert np.allclose(x.value, [2.0, 1.0], atol=1e-3)

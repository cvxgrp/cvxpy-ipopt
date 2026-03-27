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
from cvxpy.tests.base_test import BaseTest

# Check if sparsediffpy has parameter support (PR #10 bindings)
try:
    from sparsediffpy import _sparsediffengine as _de
    _de.make_parameter
    HAS_PARAM_BINDINGS = True
except AttributeError:
    HAS_PARAM_BINDINGS = False

# Check if IPOPT is available
try:
    import cyipopt  # noqa: F401
    HAS_IPOPT = True
except ImportError:
    HAS_IPOPT = False

skip_no_params = pytest.mark.skipif(
    not HAS_PARAM_BINDINGS,
    reason="sparsediffpy parameter bindings not available (install from PR #10)")
skip_no_ipopt = pytest.mark.skipif(
    not HAS_IPOPT, reason="IPOPT not installed")


@skip_no_params
@skip_no_ipopt
class TestNLPParameters(BaseTest):
    """Tests for parameter support in the DNLP diff engine."""

    def test_scalar_parameter_in_objective(self) -> None:
        """Basic test: scalar parameter in objective."""
        x = cp.Variable()
        p = cp.Parameter(value=2.0)
        prob = cp.Problem(cp.Minimize(p * x**2 + x), [x >= -5])
        prob.solve(nlp=True, solver='IPOPT')
        val1 = prob.value

        # Change parameter and re-solve (should hit fast path)
        p.value = 4.0
        prob.solve(nlp=True, solver='IPOPT')
        val2 = prob.value

        # With larger p, the quadratic term dominates more,
        # so the minimum shifts closer to 0
        assert val1 != val2
        # Analytical: min of p*x^2 + x is at x = -1/(2p), val = -1/(4p)
        self.assertAlmostEqual(val1, -1.0 / (4 * 2.0), places=4)
        self.assertAlmostEqual(val2, -1.0 / (4 * 4.0), places=4)

    def test_vector_parameter_in_objective(self) -> None:
        """Vector parameter as linear coefficient."""
        x = cp.Variable(2)
        p = cp.Parameter(2, value=[1.0, 2.0])
        prob = cp.Problem(cp.Minimize(p @ x), [x >= 0, cp.sum(x) == 1])
        prob.solve(nlp=True, solver='IPOPT')

        # With p = [1, 2], optimal is x = [1, 0], val = 1
        self.assertAlmostEqual(prob.value, 1.0, places=4)

        # Change parameter
        p.value = [3.0, 1.0]
        prob.solve(nlp=True, solver='IPOPT')

        # With p = [3, 1], optimal is x = [0, 1], val = 1
        self.assertAlmostEqual(prob.value, 1.0, places=4)

    def test_matrix_parameter_in_matmul(self) -> None:
        """Matrix parameter in A @ x."""
        A = cp.Parameter((2, 2), value=np.eye(2))
        x = cp.Variable(2)
        b = np.array([1.0, 2.0])
        # min ||Ax - b||^2
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(A @ x - b)),
            [x >= -10, x <= 10])
        prob.solve(nlp=True, solver='IPOPT')

        # With A = I, solution is x = b
        self.assertItemsAlmostEqual(x.value, b, places=3)

        # Change A to a rotation-like matrix
        A.value = np.array([[0.0, 1.0], [1.0, 0.0]])
        prob.solve(nlp=True, solver='IPOPT')

        # With A = swap matrix, solution is x = [2, 1]
        self.assertItemsAlmostEqual(x.value, [2.0, 1.0], places=3)

    def test_parameter_cache_reuse(self) -> None:
        """Verify that _nlp_cache is populated and reused."""
        x = cp.Variable()
        p = cp.Parameter(value=1.0)
        prob = cp.Problem(cp.Minimize(p * x**2 + x), [x >= -5])

        # First solve: no cache
        assert prob._nlp_cache is None
        prob.solve(nlp=True, solver='IPOPT')

        # Cache should now be populated
        assert prob._nlp_cache is not None
        assert 'solver_cache' in prob._nlp_cache
        assert 'oracles' in prob._nlp_cache['solver_cache']

        # Second solve: should reuse cache
        cached_oracles = prob._nlp_cache['solver_cache']['oracles']
        p.value = 2.0
        prob.solve(nlp=True, solver='IPOPT')

        # Same oracles object should be reused
        assert prob._nlp_cache['solver_cache']['oracles'] is cached_oracles

    def test_no_cache_without_parameters(self) -> None:
        """Problems without parameters should not create _nlp_cache."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x**2 + x), [x >= -5])
        prob.solve(nlp=True, solver='IPOPT')
        assert prob._nlp_cache is None

    def test_parameter_correctness_vs_constant(self) -> None:
        """Compare parametric solve vs constant solve for correctness."""
        np.random.seed(42)
        n = 5
        for _ in range(5):
            c_val = np.random.randn(n)

            # Solve with parameter
            x1 = cp.Variable(n)
            p = cp.Parameter(n, value=c_val)
            prob1 = cp.Problem(
                cp.Minimize(cp.sum_squares(x1) + p @ x1),
                [x1 >= -10])
            prob1.solve(nlp=True, solver='IPOPT')

            # Solve with constant (no parameter)
            x2 = cp.Variable(n)
            prob2 = cp.Problem(
                cp.Minimize(cp.sum_squares(x2) + c_val @ x2),
                [x2 >= -10])
            prob2.solve(nlp=True, solver='IPOPT')

            self.assertAlmostEqual(prob1.value, prob2.value, places=4)
            self.assertItemsAlmostEqual(x1.value, x2.value, places=3)

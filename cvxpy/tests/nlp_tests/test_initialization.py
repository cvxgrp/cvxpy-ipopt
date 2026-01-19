"""
Copyright 2026 CVXPY Developers

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
class TestVariableAttributeInit:
    """Tests initialization of variables with attributes."""

    def test_simple_diag_variable(self):
        """Test that diagonal variables work in NLP problems."""
        n = 3
        D = cp.Variable((n, n), diag=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(cp.exp(D)))))

        # Should not crash - tests diag_vec Jacobian and value propagation
        prob.solve(solver=cp.IPOPT, nlp=True, max_iter=10)

    def test_advanced_pricing_problem(self):
        """
        Test a more complex non-convex problem from Max Schaller.
        Bounds are added to prevent unboundedness.
        """
        np.random.seed(42)
        n, N = 5, 10
        rank = 2
        D = np.random.randn(n, N)
        Pitilde = np.random.randn(n + 1, N)

        Etilde_cp = cp.Variable((n, n + 1))
        Ediag = cp.Variable((n, n), diag=True)
        B = cp.Variable((n, rank))
        C = cp.Variable((rank, n))

        problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(D, Etilde_cp @ Pitilde) - cp.exp(Etilde_cp @ Pitilde))),
            [
                Etilde_cp[:, :-1] == Ediag + B @ C,
                Ediag <= 0,
                Etilde_cp >= -10, Etilde_cp <= 10,
                B >= -5, B <= 5,
                C >= -5, C <= 5,
            ]
        )

        assert not problem.is_dcp(), "Problem should be non-DCP"

        # Set initial values
        Etilde_cp.value = np.random.randn(n, n + 1)
        Ediag.value = -np.abs(np.diag(np.random.randn(n)))
        B.value = np.random.randn(n, rank)
        C.value = np.random.randn(rank, n)

        problem.solve(solver=cp.IPOPT, nlp=True, max_iter=200)

        assert problem.status == cp.OPTIMAL, "Problem did not solve to optimality"

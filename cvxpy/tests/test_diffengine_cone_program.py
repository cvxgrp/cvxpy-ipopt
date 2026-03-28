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

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestDiffengineConeProgram(BaseTest):
    """Tests for the diffengine canonicalization backend."""

    def test_simple_lp(self) -> None:
        """Non-parametric LP: compare diffengine vs default backend."""
        x = cp.Variable(3)
        c = np.array([1.0, 2.0, 3.0])
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 1, cp.sum(x) <= 10])

        prob.solve(solver=cp.CLARABEL)
        val_default = prob.value
        x_default = x.value.copy()

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFFENGINE')
        val_de = prob.value
        x_de = x.value.copy()

        self.assertAlmostEqual(val_default, val_de, places=4)
        self.assertItemsAlmostEqual(x_default, x_de, places=4)

    def test_simple_qp(self) -> None:
        """Non-parametric QP: compare diffengine vs default backend."""
        x = cp.Variable(2)
        Q = np.array([[2.0, 0.5], [0.5, 1.0]])
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, Q) + x[0]),
            [x >= -1, x <= 2],
        )

        prob.solve(solver=cp.CLARABEL)
        val_default = prob.value
        x_default = x.value.copy()

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFFENGINE')
        val_de = prob.value
        x_de = x.value.copy()

        self.assertAlmostEqual(val_default, val_de, places=4)
        self.assertItemsAlmostEqual(x_default, x_de, places=4)

    def test_parameter_conic(self) -> None:
        """Parametric conic: solve twice with different param values."""
        x = cp.Variable(2)
        p = cp.Parameter(2)

        prob = cp.Problem(cp.Minimize(p @ x), [x >= 1])

        # First solve
        p.value = np.array([1.0, 2.0])
        prob.solve(solver=cp.CLARABEL, canon_backend='DIFFENGINE')
        val_de_1 = prob.value

        prob_ref = cp.Problem(cp.Minimize(p @ x), [x >= 1])
        prob_ref.solve(solver=cp.CLARABEL)
        val_ref_1 = prob_ref.value
        self.assertAlmostEqual(val_de_1, val_ref_1, places=4)

        # Second solve with new parameter values (should re-evaluate)
        p.value = np.array([3.0, 0.5])
        prob.solve(solver=cp.CLARABEL, canon_backend='DIFFENGINE')
        val_de_2 = prob.value

        prob_ref2 = cp.Problem(cp.Minimize(p @ x), [x >= 1])
        prob_ref2.solve(solver=cp.CLARABEL)
        val_ref_2 = prob_ref2.value
        self.assertAlmostEqual(val_de_2, val_ref_2, places=4)

        # Values should differ between the two solves
        self.assertNotAlmostEqual(val_de_1, val_de_2, places=2)

    def test_parameter_qp(self) -> None:
        """Parametric QP: solve twice with different param values."""
        x = cp.Variable(2)
        p = cp.Parameter(nonneg=True)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x)),
            [x >= p],
        )

        # First solve
        p.value = 1.0
        prob.solve(solver=cp.CLARABEL, canon_backend='DIFFENGINE')
        val_de_1 = prob.value

        prob_ref = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= p])
        prob_ref.solve(solver=cp.CLARABEL)
        val_ref_1 = prob_ref.value
        self.assertAlmostEqual(val_de_1, val_ref_1, places=4)

        # Second solve with new parameter
        p.value = 3.0
        prob.solve(solver=cp.CLARABEL, canon_backend='DIFFENGINE')
        val_de_2 = prob.value

        prob_ref2 = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= p])
        prob_ref2.solve(solver=cp.CLARABEL)
        val_ref_2 = prob_ref2.value
        self.assertAlmostEqual(val_de_2, val_ref_2, places=4)

        # p=3 should give higher objective than p=1
        self.assertGreater(val_de_2, val_de_1)

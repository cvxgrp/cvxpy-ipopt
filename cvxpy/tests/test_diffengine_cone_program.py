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
import unittest

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestLPs, StandardTestSOCPs

try:
    from sparsediffpy import _sparsediffengine  # noqa: F401
    HAS_DIFFENGINE = True
except ImportError:
    HAS_DIFFENGINE = False


@pytest.mark.skipif(not HAS_DIFFENGINE, reason="sparsediffpy not installed")
class TestDiffengineConeProgram(BaseTest):
    """Tests for the DIFFENGINE canonicalization backend."""

    BACKEND = 'DIFFENGINE'

    def _solve(self, prob, **kwargs):
        """Solve with DIFFENGINE backend via Clarabel."""
        return prob.solve(solver=cp.CLARABEL, canon_backend=self.BACKEND, **kwargs)

    def _solve_default(self, prob, **kwargs):
        """Solve with default backend for comparison."""
        return prob.solve(solver=cp.CLARABEL, **kwargs)

    def test_simple_lp(self) -> None:
        """Test a simple LP: minimize c'x s.t. x >= 1."""
        x = cp.Variable(3)
        c = np.array([1.0, 2.0, 3.0])
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 1])

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de = x.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_lp_with_equality(self) -> None:
        """Test LP with equality constraints."""
        x = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(x[0] + 2 * x[1]),
            [x[0] + x[1] == 1, x >= 0],
        )

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de = x.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_lp_matrix_constraint(self) -> None:
        """Test LP with matrix variable."""
        X = cp.Variable((2, 2))
        prob = cp.Problem(
            cp.Minimize(cp.sum(X)),
            [X >= np.eye(2)],
        )

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        X_de = X.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(X_de, X.value, places=4)

    def test_symbolic_quad_form_conversion(self) -> None:
        """Test that SymbolicQuadForm is converted by the diffengine backend."""
        from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            build_variable_dict,
            convert_expr,
        )

        x = cp.Variable(2)
        P = np.array([[2.0, 0.0], [0.0, 2.0]])
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [x >= 1])

        # Dcp2Cone with quad_obj=True produces SymbolicQuadForm
        dcp2cone = Dcp2Cone(quad_obj=True)
        new_prob, _ = dcp2cone.apply(prob)
        obj_expr = new_prob.objective.expr
        self.assertEqual(type(obj_expr).__name__, "SymbolicQuadForm")

        # Verify the diffengine converter handles it
        var_dict, n_vars = build_variable_dict(new_prob.variables())
        c_obj = convert_expr(obj_expr, var_dict, n_vars)
        self.assertIsNotNone(c_obj)

    def test_qp(self) -> None:
        """Test a simple QP: minimize x'x s.t. x >= 1."""
        x = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x) + x[0]),
            [x >= 1],
        )

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de = x.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_soc_constraint(self) -> None:
        """Test with second-order cone constraint."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0]),
            [cp.norm(x[1:], 2) <= x[0], x[0] >= 0, x[1] == 1, x[2] == 1],
        )

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de = x.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_zero_and_nonneg(self) -> None:
        """Test with mixed Zero and NonNeg constraints."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [x[0] == 2, x[1:] >= 0, x[1] + x[2] == 3],
        )

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de = x.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_infeasible(self) -> None:
        """Test that infeasible problems are detected."""
        x = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [x >= 1, x <= -1],
        )
        self._solve(prob)
        self.assertEqual(prob.status, cp.INFEASIBLE)

    def test_unbounded(self) -> None:
        """Test that unbounded problems are detected."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        self._solve(prob)
        self.assertIn(prob.status, [cp.UNBOUNDED, "infeasible_or_unbounded"])

    def test_multiple_variables(self) -> None:
        """Test with multiple separate variables."""
        x = cp.Variable(2)
        y = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + 2 * cp.sum(y)),
            [x >= 1, y >= 2, x[0] + y[0] == 5],
        )

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de, y_de = x.value.copy(), y.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=4)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)
        self.assertItemsAlmostEqual(y_de, y.value, places=4)

    def test_scalar_variable(self) -> None:
        """Test with a scalar variable."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 5])

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertAlmostEqual(val_de, 5.0, places=4)

    def test_large_lp(self) -> None:
        """Test a moderate-size LP."""
        n = 50
        np.random.seed(0)
        c = np.abs(np.random.randn(n))
        A = np.random.randn(20, n)
        b = A @ np.abs(np.random.randn(n)) + 1.0

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= 0])

        val_de = self._solve(prob)
        self.assertEqual(prob.status, cp.OPTIMAL)
        x_de = x.value.copy()

        val_default = self._solve_default(prob)
        self.assertAlmostEqual(val_de, val_default, places=3)
        self.assertItemsAlmostEqual(x_de, x.value, places=3)


@pytest.mark.skipif(not HAS_DIFFENGINE, reason="sparsediffpy not installed")
class TestDiffengineStandardLPs(BaseTest):
    """Run StandardTestLPs with the DIFFENGINE backend."""

    KWARGS = dict(solver=cp.CLARABEL, canon_backend='DIFFENGINE')

    def test_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(**self.KWARGS)

    def test_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(**self.KWARGS)

    def test_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(**self.KWARGS)

    def test_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(**self.KWARGS)

    def test_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(**self.KWARGS)

    def test_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(**self.KWARGS)

    def test_lp_6(self) -> None:
        StandardTestLPs.test_lp_6(**self.KWARGS)

    @pytest.mark.skip(reason="lp_7 requires sdpap module")
    def test_lp_7(self) -> None:
        StandardTestLPs.test_lp_7(**self.KWARGS)


@pytest.mark.skipif(not HAS_DIFFENGINE, reason="sparsediffpy not installed")
class TestDiffengineStandardSOCPs(BaseTest):
    """Run StandardTestSOCPs with the DIFFENGINE backend."""

    KWARGS = dict(solver=cp.CLARABEL, canon_backend='DIFFENGINE')

    def test_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(**self.KWARGS)

    def test_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(**self.KWARGS)

    def test_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(**self.KWARGS)

    def test_socp_3ax0(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(**self.KWARGS)

    def test_socp_3ax1(self) -> None:
        StandardTestSOCPs.test_socp_3ax1(**self.KWARGS)


if __name__ == '__main__':
    unittest.main()

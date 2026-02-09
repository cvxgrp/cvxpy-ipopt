"""Tests for the diff engine conic backend.

Verifies that the DIFF_ENGINE canon backend produces identical results
to the tensor-based SCIPY backend for LP, QP, and SOCP problems.

Copyright 2025, the CVXPY developers

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
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


def _dense(M):
    """Convert sparse matrix to dense array."""
    if sp.issparse(M):
        return M.toarray()
    return np.asarray(M)


class TestDiffEngineConic(BaseTest):
    """Tests comparing DIFF_ENGINE backend against SCIPY backend."""

    def _get_data(self, prob, backend):
        """Get problem data with a specific backend (clearing cache)."""
        prob._cache = type(prob._cache)()
        data, _, _ = prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
        return data

    def _compare_data(self, prob, atol=1e-10):
        """Compare problem data from DIFF_ENGINE vs SCIPY backends."""
        data_de = self._get_data(prob, 'DIFF_ENGINE')
        data_sp = self._get_data(prob, 'SCIPY')

        A_de = _dense(data_de['A'])
        A_sp = _dense(data_sp['A'])
        np.testing.assert_allclose(A_de, A_sp, atol=atol,
                                   err_msg="A matrices don't match")
        np.testing.assert_allclose(data_de['b'], data_sp['b'], atol=atol,
                                   err_msg="b vectors don't match")
        np.testing.assert_allclose(data_de['c'], data_sp['c'], atol=atol,
                                   err_msg="c vectors don't match")

        if 'P' in data_de and 'P' in data_sp:
            P_de = _dense(data_de['P'])
            P_sp = _dense(data_sp['P'])
            np.testing.assert_allclose(P_de, P_sp, atol=atol,
                                       err_msg="P matrices don't match")

    # ---- Matrix comparison tests ----

    def test_lp_matrix_comparison(self) -> None:
        """Compare A, b, c matrices for a simple LP."""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        A_param = cp.Parameter((3, n))
        b_param = cp.Parameter(3)
        c = np.random.randn(n)

        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b_param, x >= 0])
        A_param.value = np.random.randn(3, n)
        b_param.value = np.random.randn(3) + 10

        self._compare_data(prob)

    def test_lp_with_equality(self) -> None:
        """LP with both equality and inequality constraints."""
        np.random.seed(123)
        n = 4
        x = cp.Variable(n)
        A_param = cp.Parameter((2, n))
        b_param = cp.Parameter(2)
        c = np.random.randn(n)

        prob = cp.Problem(cp.Minimize(c @ x),
                          [A_param @ x <= b_param, cp.sum(x) == 1, x >= 0])
        A_param.value = np.random.randn(2, n)
        b_param.value = np.random.randn(2) + 10

        self._compare_data(prob)

    def test_qp_matrix_comparison(self) -> None:
        """Compare P, q, A, b matrices for a QP."""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        Q_diag = cp.Parameter(n, nonneg=True)
        A_param = cp.Parameter((3, n))
        b_param = cp.Parameter(3)
        c = np.random.randn(n)

        obj = 0.5 * cp.sum(cp.multiply(Q_diag, cp.square(x))) + c @ x
        prob = cp.Problem(cp.Minimize(obj),
                          [A_param @ x <= b_param, x >= -5, x <= 5])

        Q_diag.value = np.abs(np.random.randn(n)) + 0.1
        A_param.value = np.random.randn(3, n)
        b_param.value = np.random.randn(3) + 10

        self._compare_data(prob, atol=1e-8)

    def test_no_params(self) -> None:
        """Problem with no parameters."""
        np.random.seed(42)
        n = 3
        x = cp.Variable(n)
        A = np.random.randn(2, n)
        b = np.random.randn(2) + 5

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])
        self._compare_data(prob)

    # ---- End-to-end solve tests ----

    def test_lp_solve(self) -> None:
        """Solve an LP with both backends and compare results."""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        A_param = cp.Parameter((3, n))
        b_param = cp.Parameter(3)
        c = np.random.randn(n)

        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b_param, x >= 0])
        A_param.value = np.random.randn(3, n)
        b_param.value = np.random.randn(3) + 10

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
        val_de = prob.value
        x_de = x.value.copy()

        prob._cache = type(prob._cache)()
        prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
        val_sp = prob.value

        self.assertAlmostEqual(val_de, val_sp, places=5)
        np.testing.assert_allclose(x_de, x.value, atol=1e-5)

    def test_qp_solve(self) -> None:
        """Solve a QP with both backends and compare results."""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        Q_diag = cp.Parameter(n, nonneg=True)
        c = np.random.randn(n)

        obj = 0.5 * cp.sum(cp.multiply(Q_diag, cp.square(x))) + c @ x
        prob = cp.Problem(cp.Minimize(obj), [x >= -5, x <= 5])

        Q_diag.value = np.abs(np.random.randn(n)) + 0.1

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
        val_de = prob.value

        prob._cache = type(prob._cache)()
        prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
        val_sp = prob.value

        self.assertAlmostEqual(val_de, val_sp, places=5)

    # ---- Parametric warm-path tests ----

    def test_warm_path_lp(self) -> None:
        """Test parameter updates on the warm path for LP."""
        np.random.seed(42)
        n = 6
        x = cp.Variable(n)
        A_param = cp.Parameter((3, n))
        b_param = cp.Parameter(3)
        c = np.random.randn(n)

        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b_param, x >= 0])

        # Cold path
        A_param.value = np.random.randn(3, n)
        b_param.value = np.abs(np.random.randn(3)) + 5
        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')

        # Warm path with new parameters
        for trial in range(5):
            np.random.seed(trial * 100 + 1000)
            A_param.value = np.random.randn(3, n)
            b_param.value = np.abs(np.random.randn(3)) + 5

            prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
            val_de = prob.value

            # Compare with scipy
            A_save = A_param.value.copy()
            b_save = b_param.value.copy()
            prob._cache = type(prob._cache)()
            A_param.value = A_save
            b_param.value = b_save
            prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
            val_sp = prob.value

            np.testing.assert_allclose(val_de, val_sp, atol=1e-5,
                                      err_msg=f"Warm path trial {trial}")

            # Restore DE cache for next warm solve
            prob._cache = type(prob._cache)()
            np.random.seed(42)
            A_param.value = np.random.randn(3, n)
            b_param.value = np.abs(np.random.randn(3)) + 5
            prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')

    def test_warm_path_qp(self) -> None:
        """Test parameter updates on the warm path for QP."""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        Q_diag = cp.Parameter(n, nonneg=True)
        A_param = cp.Parameter((3, n))
        b_param = cp.Parameter(3)
        c = np.random.randn(n)

        obj = 0.5 * cp.sum(cp.multiply(Q_diag, cp.square(x))) + c @ x
        prob = cp.Problem(cp.Minimize(obj),
                          [A_param @ x <= b_param, x >= -5, x <= 5])

        # Cold path
        Q_diag.value = np.abs(np.random.randn(n)) + 0.1
        A_param.value = np.random.randn(3, n)
        b_param.value = np.abs(np.random.randn(3)) + 5
        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')

        # Warm path
        for trial in range(3):
            np.random.seed(trial * 100 + 2000)
            Q_diag.value = np.abs(np.random.randn(n)) + 0.1
            A_param.value = np.random.randn(3, n)
            b_param.value = np.abs(np.random.randn(3)) + 5

            prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
            val_de = prob.value

            Q_save = Q_diag.value.copy()
            A_save = A_param.value.copy()
            b_save = b_param.value.copy()
            prob._cache = type(prob._cache)()
            Q_diag.value = Q_save
            A_param.value = A_save
            b_param.value = b_save
            prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
            val_sp = prob.value

            np.testing.assert_allclose(val_de, val_sp, atol=1e-4,
                                      err_msg=f"QP warm path trial {trial}")

            # Restore DE cache
            prob._cache = type(prob._cache)()
            Q_diag.value = np.abs(np.random.randn(n)) + 0.1
            A_param.value = np.random.randn(3, n)
            b_param.value = np.abs(np.random.randn(3)) + 5
            prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')

    # ---- Multiple parameter update tests ----

    def test_multiple_param_updates(self) -> None:
        """Loop through many parameter configurations."""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        A_param = cp.Parameter((3, n))
        b_param = cp.Parameter(3)
        c_param = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c_param @ x),
                          [A_param @ x <= b_param, x >= 0])

        for trial in range(15):
            np.random.seed(trial + 42)
            A_param.value = np.random.randn(3, n)
            b_param.value = np.abs(np.random.randn(3)) + 5
            c_param.value = np.random.randn(n)

            # DIFF_ENGINE
            prob._cache = type(prob._cache)()
            prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
            val_de = prob.value

            # SCIPY
            prob._cache = type(prob._cache)()
            prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
            val_sp = prob.value

            np.testing.assert_allclose(
                val_de, val_sp, atol=1e-5,
                err_msg=f"Trial {trial}: DE={val_de}, SP={val_sp}")

    def test_scalar_param(self) -> None:
        """Test with a single scalar parameter."""
        np.random.seed(42)
        n = 3
        x = cp.Variable(n)
        t = cp.Parameter(nonneg=True)

        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [x >= 0, cp.sum(x) <= t])
        t.value = 10.0

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
        val_de = prob.value

        prob._cache = type(prob._cache)()
        prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
        val_sp = prob.value

        self.assertAlmostEqual(val_de, val_sp, places=5)

    def test_objective_param(self) -> None:
        """Test with parameters in the objective only."""
        np.random.seed(42)
        n = 4
        x = cp.Variable(n)
        c_param = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c_param @ x), [x >= 0, cp.sum(x) == 1])
        c_param.value = np.random.randn(n)

        self._compare_data(prob)

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
        val_de = prob.value

        prob._cache = type(prob._cache)()
        prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
        val_sp = prob.value

        self.assertAlmostEqual(val_de, val_sp, places=5)

    def test_bounds(self) -> None:
        """Test with variable bounds."""
        np.random.seed(42)
        n = 4
        x = cp.Variable(n)
        A_param = cp.Parameter((2, n))
        b_param = cp.Parameter(2)

        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [A_param @ x <= b_param, x >= -1, x <= 5])
        A_param.value = np.random.randn(2, n)
        b_param.value = np.abs(np.random.randn(2)) + 5

        prob.solve(solver=cp.CLARABEL, canon_backend='DIFF_ENGINE')
        val_de = prob.value

        prob._cache = type(prob._cache)()
        prob.solve(solver=cp.CLARABEL, canon_backend='SCIPY')
        val_sp = prob.value

        self.assertAlmostEqual(val_de, val_sp, places=5)

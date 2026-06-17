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
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestStackedPermutedDense:
    # Stress tests for the stacked permuted_dense (PD) Jacobian/Hessian path in the diff engine.
    # stacked_pd originates only at left_matmul when a dense constant multiplies a matrix variable
    # or another matrix expression, so all tests here use matrix variables / matrix expressions.

    def test_multiply_spd_spd(self):
        # A dense, B dense
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(A @ X), cp.cos(B @ Y))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_spd_right(self):
        # A dense, B dense
        np.random.seed(0)
        n, m = 5, 5
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(X @ A), cp.cos((Y @ B) @ X))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_sparse(self):
        # A dense, B sparse
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = sp.random(m, n, density=0.5, format='csr')
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(A @ X), cp.cos(B @ Y))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_sparse_two(self):
        # A dense, B sparse
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = sp.random(m, n, density=0.5, format='csr')
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(A @ (X @ X).T), cp.cos(B @ Y))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_sparse_spd(self):
        # A sparse, B dense
        np.random.seed(0)
        n, m = 5, 6
        A = sp.random(m, n, density=0.5, format='csr')
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(A @ X), cp.cos(B @ Y))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_sparse_transposed_variable(self):
        np.random.seed(0)
        n = 5
        A = np.random.rand(n, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(A @ X.T), cp.cos(X))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_sparse_transposed(self):
        np.random.seed(0)
        n = 5
        A = np.random.rand(n, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin((A @ X.T).T), cp.cos(X))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_sparse_spd_transposed(self):
        np.random.seed(0)
        n = 5
        A = np.random.rand(n, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.cos(X), cp.sin((A @ X.T).T))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_deep_composition(self):
        # A deep composition of SPD results
        np.random.seed(0)
        n = 5
        m = 10
        A = np.random.rand(m, n)
        B = sp.random(n, m, density=0.5, format='csr')
        C = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(
            cp.sin(A @ cp.cos(B @ cp.logistic(C @ (X @ Y).T))),
            cp.cos(A @ cp.cos(B @ cp.logistic((C @ Y) @ (X @ Y)))),
        )))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_plain_var(self):
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((m, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(A @ X), cp.cos(Y))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_plain_var_spd(self):
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        y = cp.Variable((m, 1), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(y), cp.cos(A @ X))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_spd_spd_right_two(self):
        # Right matmul with dense A and dense B
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(n, m)
        B = np.random.rand(n, m)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(X @ A), cp.cos(Y @ B))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_pd_sparse_right(self):
        # Right matmul with dense A and sparse B
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(n, m)
        B = sp.random(n, m, density=0.5, format='csr')
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(X @ A), cp.cos(Y @ B))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_spd_index_propagation(self):
        # Indexing into a permuted dense propagates permuted dense via index_alloc /
        # index_fill_values. Use a non-sorted index with duplicates to stress the
        # permutation path.
        np.random.seed(0)
        n, m = 5, 8
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        idx_A = [0, 2, 4, 1, 3, 0, 7]
        idx_B = [0, 4, 2, 3, 1, 0, 7]
        obj = cp.Minimize(
            cp.sum(cp.multiply(cp.sin((A @ X)[idx_A]), cp.cos((B @ Y)[idx_B])))
        )
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_pd_index_propagation_right(self):
        # Right matmul with index
        np.random.seed(0)
        n, m = 5, 8
        A = np.random.rand(n, m)
        B = np.random.rand(n, m)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        idx_A = [0, 2, 4, 1, 3, 0, 7]
        idx_B = [0, 4, 2, 3, 1, 0, 7]
        obj = cp.Minimize(
            cp.sum(cp.multiply(cp.sin((X @ A)[0, idx_A]), cp.cos((Y @ B)[1, idx_B])))
        )
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_dense_left_matmul_parameter_refresh(self):
        np.random.seed(0)
        n, m = 5, 6
        A_param = cp.Parameter((m, n))
        X = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.sin(A_param @ X)))
        prob = cp.Problem(obj)

        for scale, shift in [(1.0, 0.0), (0.5, -0.5), (2.0, 0.2)]:
            A_param.value = scale * np.random.rand(m, n) + shift
            prob.solve(nlp=True)
            checker = DerivativeChecker(prob)
            checker.run_and_assert()

    def test_dense_left_matmul_parameter_multiply(self):
        np.random.seed(0)
        n, m = 5, 6
        A_param = cp.Parameter((m, n))
        B_param = cp.Parameter((m, n))
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(
            cp.sin(A_param @ X), cp.cos(B_param @ Y)
        )))
        prob = cp.Problem(obj)

        A_param.value = np.random.rand(m, n)
        B_param.value = np.random.rand(m, n)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        # Refresh both parameters and re-solve.
        A_param.value = np.random.rand(m, n) - 0.5
        B_param.value = np.random.rand(m, n) + 0.2
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_dense_left_matmul_over_right_matmul(self):
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = np.random.rand(n, m)
        X = cp.Variable((n, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.sin(A @ (X @ B))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_dense_left_matmul_over_hstack(self):
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, n), bounds=[-1, 1])
        Z = cp.hstack([X, Y])
        obj = cp.Minimize(cp.sum(cp.sin(A @ Z)))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_kron_csc_through_multiply_hessian(self):
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = np.random.rand(n, m)
        X = cp.Variable((n, n), bounds=[-1, 1])
        Y = cp.Variable((n, m), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(
            cp.sin(A @ (X @ B)),
            cp.cos(A @ Y),
        )))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

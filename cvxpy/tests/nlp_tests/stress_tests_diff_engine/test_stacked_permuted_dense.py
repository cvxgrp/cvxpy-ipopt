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
            cp.cos(A @ cp.cos(B @ cp.logistic(C @ Y))),
        )))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

"""
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

    def test_multiply_plain_var_pd(self):
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(m, bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(y), cp.cos(A @ x))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_pd_index_propagation(self):
        # Indexing into a permuted dense propagates permuted dense via index_alloc /
        # index_fill_values. Use a non-sorted index with duplicates to stress the
        # permutation path.
        np.random.seed(0)
        n, m = 5, 8
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        idx_A = [0, 2, 4, 1, 3, 0, 7]
        idx_B = [0, 4, 2, 3, 1, 0, 7]
        obj = cp.Minimize(
            cp.sum(cp.multiply(cp.sin((A @ x)[idx_A]), cp.cos((B @ y)[idx_B])))
        )
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_pd_transpose_propagation(self):
        # Transpose of a PD result. Column-shape variables make .T non-trivial:
        # (A @ x) is (m, 1), (A @ x).T is (1, m).
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        x = cp.Variable((n, 1), bounds=[-1, 1])
        y = cp.Variable((n, 1), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin((A @ x).T), cp.cos((B @ y).T))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_pd_broadcast_propagation(self):
        # Reshape PD results to column / row vectors and let multiply broadcast.
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(
            cp.reshape(cp.sin(A @ x), (m, 1), order='F'),
            cp.reshape(cp.cos(B @ y), (1, m), order='F'),
        )))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_pd_pd_right(self):
        # Right matmul with dense A and dense B
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(n, m)
        B = np.random.rand(n, m)
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(x @ A), cp.cos(y @ B))))
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
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin(x @ A), cp.cos(y @ B))))
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
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        idx_A = [0, 2, 4, 1, 3, 0, 7]
        idx_B = [0, 4, 2, 3, 1, 0, 7]
        obj = cp.Minimize(
            cp.sum(cp.multiply(cp.sin((x @ A)[idx_A]), cp.cos((y @ B)[idx_B])))
        )
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_pd_transpose_propagation_right(self):
        # Right matmul with transpose
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(n, m)
        B = np.random.rand(n, m)
        x = cp.Variable((1, n), bounds=[-1, 1])
        y = cp.Variable((1, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(cp.sin((x @ A).T), cp.cos((y @ B).T))))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_pd_broadcast_propagation_right(self):
        # Reshape right-rooted PD results and force (m, 1) * (1, m) broadcast.
        np.random.seed(0)
        n, m = 5, 6
        A = np.random.rand(n, m)
        B = np.random.rand(n, m)
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(cp.multiply(
            cp.reshape(cp.sin(x @ A), (m, 1), order='F'),
            cp.reshape(cp.cos(y @ B), (1, m), order='F'),
        )))
        prob = cp.Problem(obj)
        prob.solve(nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    """

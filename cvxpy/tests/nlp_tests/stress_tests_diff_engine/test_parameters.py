
import numpy as np
import pytest
from scipy import sparse

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestParametersDiffEngine:

    def test_scalar_param_multiply(self):
        np.random.seed(0)
        n = 5
        a = cp.Parameter(nonneg=True)
        a.value = 2.0
        x = cp.Variable(n, bounds=[0.5, 2])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(a * x))))
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

        a.value = 3.5
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

    def test_vector_param_multiply(self):
        np.random.seed(0)
        n = 5
        a = cp.Parameter(n, nonneg=True)
        a.value = np.random.rand(n) + 0.1
        x = cp.Variable(n, bounds=[0.5, 2])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(cp.multiply(a, x)))))
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

        a.value = np.random.rand(n) + 0.5
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

    def test_dense_param_matmul(self):
        np.random.seed(0)
        n = 5
        P = cp.Parameter((n, n))
        P.value = np.random.rand(n, n)
        x = cp.Variable(n, bounds=[0.5, 2])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(P @ x))))
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

        P.value = np.random.rand(n, n) * 2
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

    def test_sparse_param_matmul(self):
        np.random.seed(0)
        n = 6
        mask = np.random.rand(n, n) > 0.6
        rows, cols = np.where(mask)
        P = cp.Parameter((n, n), sparsity=(rows, cols))
        P.value_sparse = sparse.coo_array(
            (np.random.rand(len(rows)), (rows, cols)), shape=(n, n))
        x = cp.Variable(n, bounds=[0.5, 2])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(P @ x))))
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

        P.value_sparse = sparse.coo_array(
            (np.random.rand(len(rows)) * 2, (rows, cols)), shape=(n, n))
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        DerivativeChecker(prob).run_and_assert()

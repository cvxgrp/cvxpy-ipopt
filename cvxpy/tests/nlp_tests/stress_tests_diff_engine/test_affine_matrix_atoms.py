
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineMatrixAtomsDiffEngine:
    # Stress tests for affine matrix atoms in the diff engine.		
    
    def test_one_trace(self):
        np.random.seed(0)
        X = cp.Variable((10, 10))
        A = np.random.rand(10, 10)
        obj = cp.Minimize(cp.Trace(cp.log(A@ X)))
        constr = [X >= 0.5, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_two_trace(self):
        np.random.seed(0)
        Y = cp.Variable((15, 5), bounds=[0.5, 1])
        A = np.random.rand(5, 15)
        obj = cp.Minimize(cp.Trace(A @ Y))
        constr =[]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_three_trace(self):
        np.random.seed(0)
        X = cp.Variable((20, 20), bounds=[0.5, 1])
        Y = cp.Variable((20, 20), bounds=[0, 1])
        A = np.random.rand(20, 20)
        obj = cp.Minimize(cp.Trace(cp.log(A @ X) + X @ Y))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_one_transpose(self):
        np.random.seed(0)
        n, k = 10, 5
        A = np.random.randn(n, n)
        A = A + A.T 
        V = cp.Variable((n, k))
        constraints = [V.T @ V == np.eye(k)]
        Q, R = np.linalg.qr(A)

        # find k smallest eigenvalues
        obj = cp.Minimize(cp.Trace(V.T @ A @ V))
        prob = cp.Problem(obj, constraints)
        V.value = Q[:, :k]
        prob.solve(solver=cp.IPOPT, nlp=True, least_square_init_duals='no')
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        assert np.allclose(prob.value, np.sum(np.linalg.eigvalsh(A)[:k]))

        # find k largest eigenvalues
        obj = cp.Maximize(cp.Trace(V.T @ A @ V))
        prob = cp.Problem(obj, constraints)
        V.value = Q[:, :k]
        prob.solve(solver=cp.IPOPT, nlp=True, least_square_init_duals='no')
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        assert np.allclose(prob.value, np.sum(np.linalg.eigvalsh(A)[-k:]))

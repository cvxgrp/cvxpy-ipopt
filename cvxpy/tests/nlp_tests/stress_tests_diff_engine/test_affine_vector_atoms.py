import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineDiffEngine:
		# Stress tests for affine atoms in the diff engine.
		
    def test_row_broadcast(self):
        # x is 1 x n, Y is m x n
        m, n = 3, 4
        x = cp.Variable((1, n), bounds=[-2, 2])
        Y = cp.Variable((m, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = np.full((1, n), 1.5)  # strictly inside [-2,2]
        Y.value = np.full((m, n), 0.5)  # strictly inside [-1,1]
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -2, Y = -1
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -2, atol=1e-4)
        assert np.allclose(Y.value, -1, atol=1e-4)

    def test_col_broadcast(self):
        # x is m x 1, Y is m x n
        m, n = 3, 4
        x = cp.Variable((m, 1), bounds=[-2, 2])
        Y = cp.Variable((m, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = np.full((m, 1), 1.5)
        Y.value = np.full((m, n), 0.5)
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -2, Y = -1
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -2, atol=1e-4)
        assert np.allclose(Y.value, -1, atol=1e-4)

    def test_index_stress(self):
        m, n = 3, 4
        X = cp.Variable((m, n), bounds=[-2, 2])
        expr = (cp.sum(X[0, :]) + cp.sum(X[0, :]) +
                cp.sum(X[1, :]) + cp.sum(X[:, 2]) + X[0, 1] + X[2, 2])
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj)
        X.value = np.random.rand(m, n)
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: all X at lower bound
        assert prob.status == cp.OPTIMAL
        assert np.allclose(prob.value, -34.0)

    def test_duplicate_indices(self):
        m, n = 3, 3
        X = cp.Variable((m, n), bounds=[-2, 2])
        # Use duplicate indices: X[[0,0],[1,1]] = [X[0,1], X[0,1]]
        expr = cp.sum(X[[0, 0], [1, 1]]) - 2 * X[0, 1] + cp.sum(X)
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj)
        X.value = np.random.rand(m, n)
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(X.value, -2, atol=1e-4)

    def test_promote_row(self):
        # Promote scalar to row vector
        n = 4
        x = cp.Variable(bounds=[-3, 3])
        Y = cp.Variable((1, n), bounds=[-2, 2])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = 2.0
        Y.value = np.full((1, n), 1.0)
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -3, Y = -2
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -3, atol=1e-4)
        assert np.allclose(Y.value, -2, atol=1e-4)

    def test_promote_col(self):
        # Promote scalar to column vector
        m = 4
        x = cp.Variable(bounds=[-3, 3])
        Y = cp.Variable((m, 1), bounds=[-2, 2])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = 2.0
        Y.value = np.full((m, 1), 1.0)
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -3, Y = -2
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -3, atol=1e-4)
        assert np.allclose(Y.value, -2, atol=1e-4)
    
    def test_promote_add(self):
        # Scalar x, matrix Y, with bounds set via the bounds attribute
        x = cp.Variable(bounds=[-1, 1])
        Y = cp.Variable((2, 2), bounds=[0, 2])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = 0.0
        Y.value = np.full((2, 2), 1.5)
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -1, Y = 0
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -1, atol=1e-4)
        assert np.allclose(Y.value, 0, atol=1e-4)
    
    def test_reshape(self):
        x = cp.Variable(8, bounds=[-5, 5])
        A = np.random.rand(4, 2)
        obj = cp.Minimize(cp.sum_squares(cp.reshape(x, (4, 2), order='F') - A))
        prob = cp.Problem(obj)
        x.value = np.linspace(-2, 2, 8)  
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, A.flatten(order='F'), atol=1e-4)

    def test_broadcast(self):
        np.random.seed(0)
        x = cp.Variable(8, bounds=[-5, 5])
        A = np.random.rand(8, 1)
        obj = cp.Minimize(cp.sum_squares(x - A))
        prob = cp.Problem(obj)
        x.value = np.linspace(-2, 2, 8)  
        checker = DerivativeChecker(prob)
        result = checker.run()
        assert result['objective']
        assert result['gradient']
        assert result['constraints']
        assert result['jacobian']
        assert result['hessian']
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, np.mean(A), atol=1e-4)

            
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestSumIPOPT:
    """Test solving sum problems with IPOPT."""

    def test_sum_without_axis(self):
        x = cp.Variable((2, 1))
        obj = cp.Minimize((cp.sum(x) - 3)**2)
        constr = [x <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert np.allclose(x.value, [[1.0], [1.0]])
        

    def test_sum_with_axis(self):
        """Test sum with axis parameter."""
        X = cp.Variable((2, 3))
        obj = cp.Minimize(cp.sum((cp.sum(X, axis=1) - 4)**2))
        constr = [X >= 0, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        expected = np.full((2, 3), 1)
        assert np.allclose(X.value, expected)

    def test_sum_with_other_axis(self):
        """Test sum with axis parameter."""
        X = cp.Variable((2, 3))
        obj = cp.Minimize(cp.sum((cp.sum(X, axis=0) - 4)**2))
        constr = [X >= 0, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        expected = np.full((2, 3), 1)
        assert np.allclose(X.value, expected)

    def test_sum_matrix_arg(self):
        np.random.seed(0)
        n, m, k = 40, 20, 4
        A = np.random.rand(n, k) @ np.random.rand(k, m) 
        T = cp.Variable((n, m), name='T')
        obj = cp.sum(cp.multiply(A, T))
        constraints = [T >= 1, T <= 2]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none')
        assert(np.allclose(T.value, 1))
        assert problem.status == cp.OPTIMAL
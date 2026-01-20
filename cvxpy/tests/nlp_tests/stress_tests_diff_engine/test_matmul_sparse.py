import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestMatmulDifferentFormats:
    
    def test_dense_sparse_sparse(self):
        n = 10
        A = np.random.rand(n, n)
        c = np.random.rand(n, 1)
        x = cp.Variable((n, 1), nonneg=True)
        x0 = np.random.rand(n, 1)
        b = A @ x0

        x.value = 10 * np.ones((n, 1))
        obj = cp.Minimize(c.T @ x)

        # solve problem with dense A
        constraints = [A @ x == b]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        dense_val = problem.value
        dense_sol = x.value

        x.value = 10 * np.ones((n, 1))

        # solve problem with sparse A CSR
        A_sparse = sp.csr_matrix(A)
        constraints = [A_sparse @ x == b]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        sparse_val = problem.value
        sparse_sol = x.value

        x.value = 10 * np.ones((n, 1))
        # solve problem with sparse A CSC
        A_sparse = sp.csc_matrix(A)
        constraints = [A_sparse @ x == b]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        csc_val = problem.value
        csc_sol = x.value

        assert np.allclose(dense_val, sparse_val)
        assert np.allclose(dense_val, csc_val)
        assert np.allclose(dense_sol, sparse_sol)
        assert np.allclose(dense_sol, csc_sol)


        # x is 1 x n, Y is m x n
        np.random.seed(0)
        m, n = 3, 4
        x = cp.Variable((1, n), bounds=[-2, 2])
        Y = cp.Variable((m, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = np.random.rand(1, n)  
        Y.value = np.random.rand(m, n)
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

   
            
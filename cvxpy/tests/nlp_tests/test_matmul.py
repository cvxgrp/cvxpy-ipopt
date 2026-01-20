import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestMatmul():

    def test_simple_matmul_graph_form(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        t = cp.Variable(name='t')
        X.value = np.random.rand(m, n)
        Y.value = np.random.rand(n, p)
        constraints = [t == cp.sum(cp.matmul(X, Y))]
        problem = cp.Problem(cp.Minimize(t), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)
        print("successful")
        
    def test_simple_matmul_not_graph_form(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        X.value = np.random.rand(m, n)
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(X, Y))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)
        print("successful")

    def test_matmul_with_function_right(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = np.random.rand(m, n)
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(X, cp.cos(Y)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)
        print("successful")
    def test_matmul_with_function_left(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-2, 2], name='X')
        Y = np.random.rand(n, p)
        X.value = np.random.rand(m, n)
        obj = cp.sum(cp.matmul(cp.cos(X), Y))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)
        print("successful")

    def test_matmul_with_functions_both_sides(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-2, 2], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        X.value = np.random.rand(m, n)
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(cp.cos(X), cp.sin(Y)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)
        print("successful")

    def test_sparse_matrix(self):
        n = 10
        A = np.random.rand(n, n)
        c = np.random.rand(n, 1)
        x = cp.Variable((n, 1), nonneg=True)
        x0 = np.random.rand(n, 1)
        b = A @ x0

        obj = cp.Minimize(c.T @ x)

        # solve problem with dense A
        constraints = [A @ x == b]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        dense_val = problem.value
        dense_sol = x.value

        # solve problem with sparse A CSR
        A_sparse = sp.csr_matrix(A)
        constraints = [A_sparse @ x == b]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        sparse_val = problem.value
        sparse_sol = x.value

        # solve problem with sparse A CSC
        A_sparse = sp.csc_matrix(A)
        constraints = [A_sparse @ x == b]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        csc_val = problem.value
        csc_sol = x.value

        assert np.allclose(dense_val, sparse_val)
        assert np.allclose(dense_val, csc_val)
        assert np.allclose(dense_sol, sparse_sol)
        assert np.allclose(dense_sol, csc_sol)

    # this test raises an error in derivative oracle
    #@pytest.mark.xfail(reason="derivative oracle fails on this test")
    #def test_matmul_same_variable(self):
    #    print("THIS TEST FAILS")
    #    n = 3
    #    X = cp.Variable((n, n), name='X', bounds=[-2, 2])
    #    obj = cp.sum(X @ X)
    #    problem = cp.Problem(cp.Minimize(obj))
    #    problem.solve(solver=cp.IPOPT, nlp=True)
    #    print("successful")
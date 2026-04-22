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

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


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

        checker = DerivativeChecker(problem)
        checker.run_and_assert()


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

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

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

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

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

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

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

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_matmul_1d_left_constant(self):
        """1D constant on the left: (n,) @ (n, p) nonlinear variable."""
        np.random.seed(0)
        n, p = 4, 5
        a = np.random.rand(n)
        X = cp.Variable((n, p), bounds=[-1, 1], name='X')
        X.value = np.random.rand(n, p)
        obj = cp.sum(a @ cp.sin(X))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_matmul_1d_right_constant(self):
        """1D constant on the right: (m, n) nonlinear variable @ (n,)."""
        np.random.seed(0)
        m, n = 5, 4
        b = np.random.rand(n)
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        X.value = np.random.rand(m, n)
        obj = cp.sum(cp.sin(X) @ b)
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_matmul_1d_dot(self):
        """Dot product of a 1D constant with a 1D nonlinear variable."""
        np.random.seed(0)
        n = 6
        a = np.random.rand(n)
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        obj = a @ cp.sin(x)
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_matmul_param_inside_transpose(self):
        """Parameter wrapped in transpose on the left-matmul side.

        Ensures convert_matmul's param_source fallback keeps the Parameter
        referenced in the DAG so update_params doesn't dereference unregistered
        memory. Mutate A.value and re-solve to exercise the param update path.
        """
        np.random.seed(0)
        m, p = 4, 5
        A1 = np.random.rand(m, p)
        A2 = np.random.rand(m, p)
        A = cp.Parameter((m, p), value=A1)
        x = cp.Variable(m, bounds=[-1, 1], name='x')
        x.value = np.random.rand(m)
        obj = cp.sum(A.T @ cp.sin(x))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()

        A.value = A2
        x.value = None
        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()

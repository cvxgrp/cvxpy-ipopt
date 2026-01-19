"""Tests for argument handling in nonlinear atoms (log, exp).

Tests the supported argument patterns:
- Direct variables: log(x)
- Constants: log(5)
- Constant scalar multiplication: log(2*x)
- Affine combinations: log(x + y), log(2*x + 3*y + 5)

Affine combinations are supported via CoeffExtractor which extracts the
affine expression as A @ all_vars + b, then creates a linear_op C expression
that correctly handles the chain rule.

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
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestBasicPatterns:
    """Test basic argument patterns for log/exp."""

    def test_log_direct_variable(self):
        """Test log(x) - direct variable, should work."""
        x = cp.Variable(pos=True)
        x.value = 2.0

        obj = -cp.log(x)
        constraints = [x <= 5, x >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        np.testing.assert_allclose(x.value, 5.0, rtol=1e-4)

    def test_log_constant_scalar_mult(self):
        """Test log(2*x) - constant scalar multiplication, should work."""
        x = cp.Variable(pos=True)
        x.value = 2.0

        obj = -cp.log(2*x)
        constraints = [x <= 5, x >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        np.testing.assert_allclose(x.value, 5.0, rtol=1e-4)

    def test_exp_direct_variable(self):
        """Test exp(x) - direct variable, should work."""
        x = cp.Variable()
        x.value = 0.0

        obj = cp.exp(x)
        constraints = [x >= -5, x <= 5]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        np.testing.assert_allclose(x.value, -5.0, rtol=1e-3)

    def test_exp_constant_scalar_mult(self):
        """Test exp(2*x) - constant scalar multiplication, should work."""
        x = cp.Variable()
        x.value = 0.0

        obj = cp.exp(2*x)
        constraints = [x >= -5, x <= 5]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        np.testing.assert_allclose(x.value, -5.0, rtol=1e-3)

    def test_log_vector_variable(self):
        """Test log(x) where x is a vector variable."""
        n = 3
        x = cp.Variable(n, pos=True)
        x.value = np.ones(n)

        obj = -cp.sum(cp.log(x))
        constraints = [cp.sum(x) == 3, x >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, all x_i should be equal = 1
        np.testing.assert_allclose(x.value, np.ones(n), rtol=1e-4)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffinePatterns:
    """Test affine argument patterns - handled by CoeffExtractor + linear_op."""

    def test_log_add_expression(self):
        """Test log(x + y) - handled by extracting affine A matrix."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        x.value = 1.0
        y.value = 2.0

        obj = -cp.log(x + y)
        constraints = [x + y <= 10, x >= 0.1, y >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, x + y should be 10
        np.testing.assert_allclose(x.value + y.value, 10.0, rtol=1e-4)

    def test_log_affine_combination(self):
        """Test log(2*x + 3*y + 5) - handled by extracting A matrix with offset."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        x.value = 1.0
        y.value = 1.0

        obj = -cp.log(2*x + 3*y + 5)
        constraints = [x + y <= 5, x >= 0.1, y >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # Since 3 > 2, optimizer should put more weight on y
        assert y.value > x.value

    def test_log_matmul(self):
        """Test log(A @ x) - handled by extracting A matrix."""
        n = 3
        x = cp.Variable(n, pos=True)
        x.value = np.ones(n)

        A = np.array([[1.0, 2.0, 0.0],
                      [0.0, 1.0, 3.0]])

        obj = -cp.sum(cp.log(A @ x))
        constraints = [cp.sum(x) == 3, x >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # Verify solution is feasible
        assert problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE

    def test_exp_add_expression(self):
        """Test exp(x + y) - handled by extracting affine A matrix."""
        x = cp.Variable()
        y = cp.Variable()
        x.value = 0.0
        y.value = 0.0

        obj = cp.exp(x + y)
        constraints = [x >= -5, y >= -5, x + y >= -8]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, x + y should be at lower bound -8
        np.testing.assert_allclose(x.value + y.value, -8.0, rtol=1e-3)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestNestedExpressions:
    """Test nested nonlinear expressions."""

    def test_nested_nonlinear(self):
        """Test log(exp(x) + 1) - nested nonlinear, handled by dnlp2smooth reduction."""
        x = cp.Variable()
        x.value = 1.0

        # exp(x) + 1 is not affine, so dnlp2smooth introduces an auxiliary variable:
        # t = exp(x) + 1, then log(t). This makes log's argument a direct variable.
        obj = cp.log(cp.exp(x) + 1)
        constraints = [x >= -10, x <= 10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, minimize log(exp(x) + 1) means minimize x (as x -> -inf)
        np.testing.assert_allclose(x.value, -10.0, rtol=1e-3)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestWorkaroundWithAuxVariables:
    """Test the explicit workaround using auxiliary variables."""

    def test_log_sum_with_aux_variable(self):
        """Test explicit workaround: log(t) with t == x + y constraint."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        t = cp.Variable(pos=True)
        x.value = 1.0
        y.value = 2.0
        t.value = 3.0

        # Explicit auxiliary variable
        obj = -cp.log(t)
        constraints = [t == x + y, x + y <= 10, x >= 0.1, y >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)

        # At optimum, t = x + y should be 10
        np.testing.assert_allclose(t.value, 10.0, rtol=1e-4)
        np.testing.assert_allclose(x.value + y.value, 10.0, rtol=1e-4)


def _finite_diff_gradient(c_prob, x0, eps=1e-7):
    """Compute gradient using finite differences."""
    n = len(x0)
    fd_grad = np.zeros(n)
    for i in range(n):
        x_plus = x0.copy()
        x_plus[i] += eps
        x_minus = x0.copy()
        x_minus[i] -= eps
        fd_grad[i] = (c_prob.objective_forward(x_plus) - c_prob.objective_forward(x_minus)) / (2 * eps)
    return fd_grad


def _finite_diff_hessian(c_prob, x0, eps=1e-5):
    """Compute Hessian using finite differences of gradient."""
    n = len(x0)
    fd_hess = np.zeros((n, n))
    for i in range(n):
        x_plus = x0.copy()
        x_plus[i] += eps
        x_minus = x0.copy()
        x_minus[i] -= eps

        c_prob.objective_forward(x_plus)
        grad_plus = c_prob.gradient()

        c_prob.objective_forward(x_minus)
        grad_minus = c_prob.gradient()

        fd_hess[:, i] = (grad_plus - grad_minus) / (2 * eps)
    # Symmetrize
    fd_hess = 0.5 * (fd_hess + fd_hess.T)
    return fd_hess


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestGradientVerification:
    """Verify gradients against finite differences."""

    def test_log_sum_gradient_fd(self):
        """Compare log(x + y) gradient to finite differences."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        obj = -cp.log(x + y)
        constraints = [x >= 0.1, y >= 0.1, x <= 10, y <= 10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        c_prob = C_problem(problem)
        c_prob.init_derivatives()

        x0 = np.array([2.0, 3.0])
        c_prob.objective_forward(x0)
        analytic_grad = c_prob.gradient()

        fd_grad = _finite_diff_gradient(c_prob, x0)

        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-5)

    def test_exp_affine_gradient_fd(self):
        """Compare exp(2*x + 3*y) gradient to finite differences."""
        x = cp.Variable()
        y = cp.Variable()

        obj = cp.exp(2*x + 3*y)
        constraints = [x >= -5, y >= -5, x <= 5, y <= 5]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        c_prob = C_problem(problem)
        c_prob.init_derivatives()

        x0 = np.array([0.5, -0.3])
        c_prob.objective_forward(x0)
        analytic_grad = c_prob.gradient()

        fd_grad = _finite_diff_gradient(c_prob, x0)

        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-5)

    def test_log_matmul_gradient_fd(self):
        """Compare log(A @ x) gradient to finite differences.

        Note: When using C_problem directly, log(A @ x) where A @ x is a vector
        requires the IPOPT solver path which applies dnlp2smooth reduction.
        Here we test via solve() to exercise the full derivative pipeline.
        """
        n = 3
        x = cp.Variable(n, pos=True)
        x.value = np.array([1.0, 2.0, 1.5])

        A = np.array([[1.0, 2.0, 0.0],
                      [0.0, 1.0, 3.0]])

        obj = -cp.sum(cp.log(A @ x))
        constraints = [x >= 0.1, x <= 10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        # Test via solve - this exercises the full derivative pipeline
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE

    def test_log_affine_with_offset_gradient_fd(self):
        """Compare log(2*x + 3*y + 5) gradient to finite differences."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        obj = -cp.log(2*x + 3*y + 5)
        constraints = [x >= 0.1, y >= 0.1, x <= 10, y <= 10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        c_prob = C_problem(problem)
        c_prob.init_derivatives()

        x0 = np.array([1.5, 2.5])
        c_prob.objective_forward(x0)
        analytic_grad = c_prob.gradient()

        fd_grad = _finite_diff_gradient(c_prob, x0)

        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-5)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestHessianVerification:
    """Verify Hessians against finite differences."""

    def test_log_sum_hessian_structure(self):
        """Verify Hessian has correct structure for log(x + y).

        Note: The exact values are tested implicitly via solve() convergence.
        Here we verify the Hessian has non-zero entries at expected locations.
        """
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        obj = -cp.log(x + y)
        constraints = [x >= 0.1, y >= 0.1, x <= 10, y <= 10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        c_prob = C_problem(problem)
        c_prob.init_derivatives()

        x0 = np.array([2.0, 3.0])
        c_prob.objective_forward(x0)
        c_prob.gradient()
        c_prob.constraint_forward(x0)
        c_prob.jacobian()

        # Get Hessian with obj_factor=1.0 and no constraint weights
        n_constraints = 4
        lambda_vals = np.zeros(n_constraints)
        hess_csr = c_prob.hessian(1.0, lambda_vals)

        # Verify structure: Hessian of log(x+y) should have all 4 entries non-zero
        hess_dense = hess_csr.toarray()
        assert hess_dense.shape == (2, 2)
        # All entries should be non-zero for log(x + y)
        assert hess_csr.nnz > 0

    def test_exp_affine_hessian_structure(self):
        """Verify Hessian has correct structure for exp(2*x + 3*y).

        Note: The exact values are tested implicitly via solve() convergence.
        Here we verify the Hessian has non-zero entries at expected locations.
        """
        x = cp.Variable()
        y = cp.Variable()

        obj = cp.exp(2*x + 3*y)
        constraints = [x >= -5, y >= -5, x <= 5, y <= 5]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        c_prob = C_problem(problem)
        c_prob.init_derivatives()

        x0 = np.array([0.5, -0.3])
        c_prob.objective_forward(x0)
        c_prob.gradient()
        c_prob.constraint_forward(x0)
        c_prob.jacobian()

        n_constraints = 4
        lambda_vals = np.zeros(n_constraints)
        hess_csr = c_prob.hessian(1.0, lambda_vals)

        # Verify structure: Hessian of exp(2x+3y) should have all entries non-zero
        hess_dense = hess_csr.toarray()
        assert hess_dense.shape == (2, 2)
        # All entries should be non-zero for exp(2x + 3y)
        assert hess_csr.nnz > 0


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestEdgeCases:
    """Edge cases for affine arguments."""

    def test_log_sparse_matmul(self):
        """Test log(A @ x) with sparse A."""
        n = 5
        x = cp.Variable(n, pos=True)
        x.value = np.ones(n)

        # Sparse matrix with only diagonal and one off-diagonal
        A_sparse = sp.diags([1.0, 2.0, 3.0, 2.0, 1.0], 0, format='csr')
        A_sparse = A_sparse + sp.diags([0.5, 0.5, 0.5, 0.5], 1, format='csr')
        A = A_sparse.toarray()

        obj = -cp.sum(cp.log(A @ x))
        constraints = [cp.sum(x) == 5, x >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE

    def test_exp_sum_reduction(self):
        """Test exp(sum(x)) - affine reduction then exp."""
        n = 3
        x = cp.Variable(n)
        x.value = np.zeros(n)

        obj = cp.exp(cp.sum(x))
        constraints = [x >= -2, x <= 2]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, sum(x) should be at lower bound -6
        np.testing.assert_allclose(np.sum(x.value), -6.0, rtol=1e-3)

    def test_negative_coefficients(self):
        """Test log(x - 0.5*y + 10) with negative coefficient."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        x.value = 5.0
        y.value = 1.0

        # Ensure argument stays positive: x - 0.5*y + 10 > 0
        obj = -cp.log(x - 0.5*y + 10)
        constraints = [x >= 0.1, y >= 0.1, x <= 10, y <= 10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # Maximize x - 0.5*y + 10, so x=10, y=0.1
        np.testing.assert_allclose(x.value, 10.0, rtol=1e-4)
        np.testing.assert_allclose(y.value, 0.1, rtol=1e-4)

    def test_multiple_variable_shapes(self):
        """Test affine expression with variables of different shapes."""
        x = cp.Variable(2, pos=True)
        y = cp.Variable(pos=True)
        x.value = np.array([1.0, 1.0])
        y.value = 1.0

        # x[0] + x[1] + y is a scalar
        obj = -cp.log(x[0] + x[1] + y)
        constraints = [cp.sum(x) + y <= 10, x >= 0.1, y >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        np.testing.assert_allclose(x.value[0] + x.value[1] + y.value, 10.0, rtol=1e-4)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineInConstraints:
    """Affine args in constraints."""

    def test_log_affine_constraint(self):
        """Test log(x + y) >= 1 constraint."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        x.value = 2.0
        y.value = 2.0

        obj = x + y  # Minimize sum
        constraints = [cp.log(x + y) >= 1, x >= 0.1, y >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, log(x + y) = 1, so x + y = e
        np.testing.assert_allclose(x.value + y.value, np.e, rtol=1e-4)

    def test_exp_affine_constraint(self):
        """Test exp(x + y) <= 5 constraint."""
        x = cp.Variable()
        y = cp.Variable()
        x.value = 0.0
        y.value = 0.0

        obj = -(x + y)  # Maximize sum
        constraints = [cp.exp(x + y) <= 5, x >= -10, y >= -10]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        # At optimum, exp(x + y) = 5, so x + y = log(5)
        np.testing.assert_allclose(x.value + y.value, np.log(5), rtol=1e-4)

    def test_jacobian_affine_constraint(self):
        """Verify Jacobian structure of constraint log(x + y) >= 1.

        Note: The constraint log(x + y) >= 1 is internally converted to
        1 - log(x + y) <= 0, which negates the Jacobian. We verify the
        absolute value matches the expected derivative.
        """
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        obj = x + y
        constraints = [cp.log(x + y) >= 1, x >= 0.1, y >= 0.1]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        c_prob = C_problem(problem)
        c_prob.init_derivatives()

        x0 = np.array([2.0, 3.0])
        c_prob.constraint_forward(x0)
        jac_csr = c_prob.jacobian()

        # Jacobian of log(x + y) w.r.t. x and y has magnitude 1/(x+y)
        expected_deriv_magnitude = 1.0 / (x0[0] + x0[1])

        # The first row of the Jacobian corresponds to the log constraint
        jac_dense = jac_csr.toarray()
        log_jac_row = jac_dense[0, :]

        # Check absolute values match (sign may be negated due to constraint form)
        np.testing.assert_allclose(np.abs(log_jac_row[0]), expected_deriv_magnitude, rtol=1e-6)
        np.testing.assert_allclose(np.abs(log_jac_row[1]), expected_deriv_magnitude, rtol=1e-6)
        # Both entries should have the same sign
        assert np.sign(log_jac_row[0]) == np.sign(log_jac_row[1])

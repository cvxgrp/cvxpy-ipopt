"""Tests for fast parameter support in the diff engine.

Verifies that CVXPY Parameters are correctly handled as live nodes in the
C expression tree, allowing parameter updates without tree rebuilding.

All tests use **affine** parameter expressions only — matching what DCP2Cone
produces.  After canonicalization the diff engine sees c^T x objectives and
Ax + b constraints where c, A, b depend on parameters.

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
from scipy import sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestFastParams:

    # ------------------------------------------------------------------
    #  Parameter node creation and forward pass
    # ------------------------------------------------------------------
    def test_parameter_node_forward(self):
        """Parameter node value updates correctly via update_params."""
        P = cp.Parameter((2, 2))
        x = cp.Variable(2, bounds=[-10, 10])
        P.value = np.eye(2)
        x.value = np.array([1.0, 2.0])

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 2.0])
        val = c_prob.objective_forward(u)
        np.testing.assert_allclose(val, 3.0)  # sum(I @ [1,2]) = 3

        # Update parameter to 2*I
        P.value = 2 * np.eye(2)
        c_prob.update_params()
        val = c_prob.objective_forward(u)
        np.testing.assert_allclose(val, 6.0)  # sum(2I @ [1,2]) = 6

    # ------------------------------------------------------------------
    #  P @ x — Parameter matrix × variable vector
    # ------------------------------------------------------------------
    def test_param_matmul_gradient(self):
        """P @ x: gradient = P.T @ 1 for sum objective."""
        P = cp.Parameter((3, 3))
        x = cp.Variable(3, bounds=[-10, 10])

        P.value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        x.value = np.ones(3)

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 2.0, 3.0])
        c_prob.objective_forward(u)
        grad = c_prob.gradient()

        expected = P.value.T @ np.ones(3)  # [12, 15, 18]
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_param_matmul_update(self):
        """After updating P, gradient changes without tree rebuild."""
        P = cp.Parameter((3, 3))
        x = cp.Variable(3, bounds=[-10, 10])
        x.value = np.ones(3)

        # First: P = I
        P.value = np.eye(3)
        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 2.0, 3.0])
        c_prob.objective_forward(u)
        grad1 = c_prob.gradient()
        np.testing.assert_allclose(grad1, [1.0, 1.0, 1.0], atol=1e-10)

        # Update P to [[1,2,3],[4,5,6],[7,8,9]]
        P.value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        c_prob.update_params()  # NO rebuild
        c_prob.objective_forward(u)
        grad2 = c_prob.gradient()
        expected = P.value.T @ np.ones(3)
        np.testing.assert_allclose(grad2, expected, atol=1e-10)

    def test_param_matmul_finite_diff(self):
        """P @ x: verify gradient with central differences."""
        np.random.seed(42)
        P = cp.Parameter((3, 4))
        x = cp.Variable(4, bounds=[-10, 10])

        P.value = np.random.randn(3, 4)
        x.value = np.random.randn(4)

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.random.randn(4)
        c_prob.objective_forward(u)
        grad = c_prob.gradient()

        eps = 1e-7
        for j in range(4):
            u_p, u_m = u.copy(), u.copy()
            u_p[j] += eps
            u_m[j] -= eps
            numerical = (c_prob.objective_forward(u_p) - c_prob.objective_forward(u_m)) / (2 * eps)
            np.testing.assert_allclose(grad[j], numerical, atol=1e-5,
                                       err_msg=f"Gradient mismatch at index {j}")

    # ------------------------------------------------------------------
    #  gamma * expr — Scalar parameter × affine expression
    # ------------------------------------------------------------------
    def test_param_scalar_mult_gradient(self):
        """gamma * sum(x): gradient = gamma * 1."""
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(4, bounds=[-10, 10])

        gamma.value = 2.0
        x.value = np.array([1.0, 2.0, 3.0, 4.0])

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 2.0, 3.0, 4.0])
        c_prob.objective_forward(u)
        grad = c_prob.gradient()
        np.testing.assert_allclose(grad, 2.0 * np.ones(4), atol=1e-10)

        # Update gamma
        gamma.value = 5.0
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad = c_prob.gradient()
        np.testing.assert_allclose(grad, 5.0 * np.ones(4), atol=1e-10)

    def test_param_scalar_mult_hessian(self):
        """gamma * sum(x): Hessian = 0 (affine objective)."""
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(4, bounds=[-10, 10])

        gamma.value = 5.0
        x.value = np.ones(4)

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 2.0, 3.0, 4.0])
        c_prob.objective_forward(u)
        c_prob.constraint_forward(u)
        hess = c_prob.hessian(1.0, np.array([]))
        np.testing.assert_allclose(hess.toarray(), np.zeros((4, 4)), atol=1e-10)

    # ------------------------------------------------------------------
    #  p ∘ x — Vector parameter × variable
    # ------------------------------------------------------------------
    def test_param_vector_mult(self):
        """p ∘ x: gradient = p for sum(p * x)."""
        p = cp.Parameter(4, pos=True)
        x = cp.Variable(4, bounds=[-10, 10])

        p.value = np.array([1.0, 2.0, 3.0, 4.0])
        x.value = np.ones(4)

        prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(p, x))))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 1.0, 1.0, 1.0])
        c_prob.objective_forward(u)
        grad = c_prob.gradient()
        np.testing.assert_allclose(grad, p.value, atol=1e-10)

        # Update p
        p.value = np.array([10.0, 20.0, 30.0, 40.0])
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad = c_prob.gradient()
        np.testing.assert_allclose(grad, p.value, atol=1e-10)

    # ------------------------------------------------------------------
    #  Affine parametric objective: sum(A @ x - b)
    # ------------------------------------------------------------------
    def test_param_affine_objective(self):
        """sum(A @ x - b): gradient = A.T @ 1, Hessian = 0."""
        np.random.seed(0)
        n = 3
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        x = cp.Variable(n, bounds=[-10, 10])

        A.value = np.random.randn(n, n)
        b.value = np.random.randn(n)
        x.value = np.random.randn(n)

        prob = cp.Problem(cp.Minimize(cp.sum(A @ x - b)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.random.randn(n)
        c_prob.objective_forward(u)
        grad = c_prob.gradient()

        # Analytical: gradient of sum(A @ x - b) w.r.t. x = A.T @ 1
        Av = A.value
        expected_grad = Av.T @ np.ones(n)
        np.testing.assert_allclose(grad, expected_grad, atol=1e-8)

        # Hessian = 0 (affine)
        c_prob.constraint_forward(u)
        hess = c_prob.hessian(1.0, np.array([]))
        np.testing.assert_allclose(hess.toarray(), np.zeros((n, n)), atol=1e-8)

        # Update A and b, check again
        A.value = np.eye(n) * 3
        b.value = np.ones(n) * 2
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad2 = c_prob.gradient()
        expected_grad2 = (3 * np.eye(n)).T @ np.ones(n)
        np.testing.assert_allclose(grad2, expected_grad2, atol=1e-8)

    # ------------------------------------------------------------------
    #  Finite difference checks for gradient/Jacobian/Hessian
    # ------------------------------------------------------------------
    def test_param_finite_diff_gradient(self):
        """Central differences for objective gradient with parameters."""
        np.random.seed(123)
        gamma = cp.Parameter(nonneg=True)
        P = cp.Parameter((3, 3))
        x = cp.Variable(3, bounds=[-5, 5])

        gamma.value = 2.5
        P.value = np.random.randn(3, 3)
        x.value = np.random.randn(3)

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.random.randn(3) * 0.5
        c_prob.objective_forward(u)
        grad = c_prob.gradient()

        eps = 1e-7
        for j in range(3):
            u_p, u_m = u.copy(), u.copy()
            u_p[j] += eps
            u_m[j] -= eps
            numerical = (c_prob.objective_forward(u_p) - c_prob.objective_forward(u_m)) / (2 * eps)
            np.testing.assert_allclose(grad[j], numerical, atol=1e-5,
                                       err_msg=f"Gradient mismatch at index {j}")

    def test_param_finite_diff_jacobian(self):
        """Central differences for constraint Jacobian with parameters."""
        np.random.seed(456)
        A = cp.Parameter((3, 4))
        b = cp.Parameter(3)
        x = cp.Variable(4, bounds=[-5, 5])

        A.value = np.random.randn(3, 4)
        b.value = np.random.randn(3)
        x.value = np.random.randn(4)

        prob = cp.Problem(cp.Minimize(0), [A @ x - b <= 0])
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.random.randn(4) * 0.5
        c_prob.constraint_forward(u)
        jac = c_prob.jacobian().toarray()

        eps = 1e-7
        n_vars = 4
        n_constraints = 3
        numerical_jac = np.zeros((n_constraints, n_vars))
        for j in range(n_vars):
            u_p, u_m = u.copy(), u.copy()
            u_p[j] += eps
            u_m[j] -= eps
            c_p = c_prob.constraint_forward(u_p)
            c_m = c_prob.constraint_forward(u_m)
            numerical_jac[:, j] = (c_p - c_m) / (2 * eps)

        np.testing.assert_allclose(jac, numerical_jac, atol=1e-5)

    def test_param_finite_diff_hessian(self):
        """Central differences for Lagrangian Hessian with parameters (affine = 0)."""
        np.random.seed(789)
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(3, bounds=[-5, 5])

        gamma.value = 3.0
        x.value = np.random.randn(3)

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.random.randn(3) * 0.5
        c_prob.objective_forward(u)
        c_prob.constraint_forward(u)
        hess = c_prob.hessian(1.0, np.array([])).toarray()

        # Affine objective → Hessian is zero
        np.testing.assert_allclose(hess, np.zeros((3, 3)), atol=1e-10)

        # Also verify via finite differences of the gradient
        eps = 1e-5
        n_vars = 3
        numerical_hess = np.zeros((n_vars, n_vars))
        for j in range(n_vars):
            u_p, u_m = u.copy(), u.copy()
            u_p[j] += eps
            u_m[j] -= eps
            c_prob.objective_forward(u_p)
            grad_p = c_prob.gradient().copy()
            c_prob.objective_forward(u_m)
            grad_m = c_prob.gradient().copy()
            numerical_hess[:, j] = (grad_p - grad_m) / (2 * eps)

        np.testing.assert_allclose(hess, numerical_hess, atol=1e-4)

    # ------------------------------------------------------------------
    #  No-rebuild performance: update_params doesn't require re-init
    # ------------------------------------------------------------------
    def test_no_rebuild(self):
        """Verify update_params + re-eval works without init_jacobian/init_hessian."""
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(3, bounds=[-10, 10])
        gamma.value = 1.0
        x.value = np.ones(3)

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.array([1.0, 2.0, 3.0])

        # First evaluation
        c_prob.objective_forward(u)
        grad1 = c_prob.gradient()

        # Update parameter — NO re-init of structures
        gamma.value = 10.0
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad2 = c_prob.gradient()

        np.testing.assert_allclose(grad1, np.ones(3), atol=1e-10)
        np.testing.assert_allclose(grad2, 10.0 * np.ones(3), atol=1e-10)

    # ------------------------------------------------------------------
    #  Finance-style: affine parametric objective
    # ------------------------------------------------------------------
    def test_param_finance(self):
        """Affine finance-style: gamma * (c.T @ x) - mu @ x."""
        np.random.seed(99)
        n = 5
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(n, bounds=[0, 1])

        c_vec = np.random.randn(n)
        mu = np.random.randn(n)

        gamma.value = 1.0
        x.value = np.ones(n) / n

        obj = gamma * (c_vec @ x) - mu @ x
        prob = cp.Problem(cp.Minimize(obj), [cp.sum(x) == 1])
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()
        c_prob.update_params()

        u = np.ones(n) / n
        c_prob.objective_forward(u)
        grad = c_prob.gradient()

        # Analytical: gradient = gamma * c - mu
        expected = gamma.value * c_vec - mu
        np.testing.assert_allclose(grad, expected, atol=1e-8)

        # Update gamma, check gradient changes
        gamma.value = 5.0
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad2 = c_prob.gradient()
        expected2 = 5.0 * c_vec - mu
        np.testing.assert_allclose(grad2, expected2, atol=1e-8)

    # ------------------------------------------------------------------
    #  Problem with no parameters (backward compatibility)
    # ------------------------------------------------------------------
    def test_no_params_backward_compat(self):
        """Problems without parameters still work normally (affine)."""
        x = cp.Variable(3, bounds=[-10, 10])
        x.value = np.ones(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()

        u = np.array([1.0, 2.0, 3.0])
        c_prob.objective_forward(u)
        grad = c_prob.gradient()
        np.testing.assert_allclose(grad, np.ones(3), atol=1e-10)

    # ------------------------------------------------------------------
    #  DerivativeChecker with parameters
    # ------------------------------------------------------------------
    def test_derivative_checker_with_params(self):
        """Run DerivativeChecker on a parametric affine problem."""
        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker

        np.random.seed(42)
        P = cp.Parameter((3, 3))
        x = cp.Variable(3, bounds=[-5, 5])

        P.value = np.random.randn(3, 3)
        x.value = np.random.randn(3)

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_derivative_checker_scalar_param(self):
        """Run DerivativeChecker on gamma * sum(x)."""
        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker

        np.random.seed(42)
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(4, bounds=[-5, 5])

        gamma.value = 3.0
        x.value = np.random.randn(4)

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(x)))
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_derivative_checker_vector_param(self):
        """Run DerivativeChecker on sum(p * x)."""
        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker

        np.random.seed(42)
        p = cp.Parameter(4, pos=True)
        x = cp.Variable(4, bounds=[-5, 5])

        p.value = np.abs(np.random.randn(4)) + 0.1
        x.value = np.random.randn(4)

        prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(p, x))))
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    # ------------------------------------------------------------------
    #  Multiple parameter updates
    # ------------------------------------------------------------------
    def test_multiple_param_updates(self):
        """Multiple update_params calls with different values."""
        gamma = cp.Parameter(nonneg=True)
        x = cp.Variable(2, bounds=[-10, 10])
        gamma.value = 1.0
        x.value = np.ones(2)

        prob = cp.Problem(cp.Minimize(gamma * cp.sum(x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()

        u = np.array([1.0, 2.0])
        for gamma_val in [1.0, 2.0, 0.5, 10.0, 0.1]:
            gamma.value = gamma_val
            c_prob.update_params()
            c_prob.objective_forward(u)
            grad = c_prob.gradient()
            np.testing.assert_allclose(grad, gamma_val * np.ones(2), atol=1e-10,
                                       err_msg=f"Failed for gamma={gamma_val}")

    # ------------------------------------------------------------------
    #  Sparse parameters (sparsity kwarg)
    # ------------------------------------------------------------------
    def test_sparse_param_matmul_forward(self):
        """Sparse P @ x: forward pass with dense sparsity pattern."""
        sparsity = ([0, 0, 1, 1], [0, 1, 0, 1])
        P = cp.Parameter((2, 2), sparsity=sparsity)
        x = cp.Variable(2, bounds=[-10, 10])

        P.value_sparse = sp.coo_array(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        x.value = np.array([1.0, 2.0])

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()

        u = np.array([1.0, 2.0])
        val = c_prob.objective_forward(u)
        # P @ [1,2] = [5, 11], sum = 16
        np.testing.assert_allclose(val, 16.0)

    def test_sparse_param_matmul_gradient(self):
        """Sparse P @ x: gradient = P.T @ 1 for sum objective."""
        sparsity = ([0, 0, 1, 1], [0, 1, 0, 1])
        P = cp.Parameter((2, 2), sparsity=sparsity)
        x = cp.Variable(2, bounds=[-10, 10])

        P.value_sparse = sp.coo_array(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        x.value = np.ones(2)

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()

        u = np.array([1.0, 2.0])
        c_prob.objective_forward(u)
        grad = c_prob.gradient()

        Pv = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = Pv.T @ np.ones(2)  # [4, 6]
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_sparse_param_matmul_update(self):
        """After updating sparse P, gradient changes without tree rebuild."""
        sparsity = ([0, 0, 1, 1], [0, 1, 0, 1])
        P = cp.Parameter((2, 2), sparsity=sparsity)
        x = cp.Variable(2, bounds=[-10, 10])
        x.value = np.ones(2)

        P.value_sparse = sp.coo_array(
            ([1.0, 0.0, 0.0, 1.0], ([0, 0, 1, 1], [0, 1, 0, 1])), shape=(2, 2)
        )
        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()

        u = np.array([1.0, 2.0])
        c_prob.objective_forward(u)
        grad1 = c_prob.gradient()
        np.testing.assert_allclose(grad1, [1.0, 1.0], atol=1e-10)

        # Update P via value_sparse
        P.value_sparse = sp.coo_array(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad2 = c_prob.gradient()
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]).T @ np.ones(2)
        np.testing.assert_allclose(grad2, expected, atol=1e-10)

    def test_sparse_param_truly_sparse(self):
        """Sparse parameter with actual zeros (diagonal pattern)."""
        sparsity = ([0, 1, 2], [0, 1, 2])  # diagonal only
        P = cp.Parameter((3, 3), sparsity=sparsity)
        x = cp.Variable(3, bounds=[-10, 10])

        P.value_sparse = sp.coo_array(
            ([2.0, 3.0, 4.0], ([0, 1, 2], [0, 1, 2])), shape=(3, 3)
        )
        x.value = np.ones(3)

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        c_prob = C_problem(prob, verbose=False)
        c_prob.init_jacobian()
        c_prob.init_hessian()

        u = np.array([1.0, 2.0, 3.0])
        val = c_prob.objective_forward(u)
        # diag([2,3,4]) @ [1,2,3] = [2,6,12], sum = 20
        np.testing.assert_allclose(val, 20.0)

        c_prob.objective_forward(u)
        grad = c_prob.gradient()
        # grad of sum(diag(p) @ x) = p (diagonal values)
        np.testing.assert_allclose(grad, [2.0, 3.0, 4.0], atol=1e-10)

        # Update diagonal values
        P.value_sparse = sp.coo_array(
            ([10.0, 20.0, 30.0], ([0, 1, 2], [0, 1, 2])), shape=(3, 3)
        )
        c_prob.update_params()
        c_prob.objective_forward(u)
        grad2 = c_prob.gradient()
        np.testing.assert_allclose(grad2, [10.0, 20.0, 30.0], atol=1e-10)

    def test_sparse_param_derivative_checker(self):
        """Run DerivativeChecker on a sparse parametric problem."""
        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker

        np.random.seed(42)
        sparsity = ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2])
        P = cp.Parameter((3, 3), sparsity=sparsity)
        x = cp.Variable(3, bounds=[-5, 5])

        vals = np.random.randn(6)
        P.value_sparse = sp.coo_array(
            (vals, ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2])), shape=(3, 3)
        )
        x.value = np.random.randn(3)

        prob = cp.Problem(cp.Minimize(cp.sum(P @ x)))
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

"""Integration tests for affine atoms with NLP solvers.

Tests that affine atoms (trace, diag, upper_tri, cumsum, convolve, concatenate, wrap)
work correctly in actual optimization problems using IPOPT.
"""
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.affine.conv import convolve
from cvxpy.atoms.affine.wraps import nonneg_wrap
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestTraceIntegration:
    """Integration tests for trace atom with NLP solver."""

    def test_trace_minimize_log_det_proxy(self):
        """Minimize trace(log(X)) as a proxy for log-determinant."""
        X = cp.Variable((3, 3), bounds=[0.1, 10])
        # Minimize trace(log(X)) subject to X being element-wise bounded
        # At optimum, diagonal elements should be at lower bound
        prob = cp.Problem(cp.Minimize(cp.trace(cp.log(X))),
                          [cp.sum(X) >= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        # The solution should have small diagonal elements
        assert np.all(np.diag(X.value) < 1.0)

    def test_trace_maximize_with_constraint(self):
        """Maximize trace(exp(X)) with sum constraint."""
        X = cp.Variable((2, 2), bounds=[-2, 2])
        prob = cp.Problem(cp.Maximize(cp.trace(cp.exp(X))),
                          [cp.sum(X) <= 4])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        # Diagonal elements should be maximized
        assert np.diag(X.value).sum() > 0

    def test_trace_in_constraint(self):
        """Use trace in a constraint."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        prob = cp.Problem(cp.Minimize(cp.sum(X)),
                          [cp.trace(cp.log(X)) >= -1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestDiagVecIntegration:
    """Integration tests for diag_vec atom (vector to diagonal matrix)."""

    def test_diag_vec_minimize_sum_exp(self):
        """Minimize sum of exp of diagonal matrix entries."""
        v = cp.Variable(3, bounds=[-2, 2])
        D = cp.diag(v)  # 3x3 diagonal matrix
        prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(D))),
                          [cp.sum(v) >= 0])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        # Off-diagonal elements are zero, so exp(0) = 1
        # Diagonal elements should be minimized

    def test_diag_vec_with_exp(self):
        """Use diag_vec with nonlinear function on diagonal."""
        v = cp.Variable(3, bounds=[0.1, 5])
        D = cp.diag(v)  # 3x3 diagonal matrix
        # Minimize sum of exp of diagonal elements via trace
        prob = cp.Problem(cp.Minimize(cp.trace(cp.exp(D))),
                          [cp.sum(v) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        # Solution should have small diagonal elements (close to lower bound)
        # trace(exp(D)) = sum(exp(diag)) + 6*exp(0) = sum(exp(v)) + 6
        # minimized when v is at lower bound


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestDiagMatIntegration:
    """Integration tests for diag_mat atom (extract diagonal from matrix)."""

    def test_diag_mat_minimize_log(self):
        """Minimize sum of log of diagonal elements."""
        X = cp.Variable((3, 3), bounds=[0.1, 10])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.diag(X)))),
                          [cp.sum(X) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_diag_mat_in_constraint(self):
        """Use diag(X) in constraint."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        prob = cp.Problem(cp.Minimize(cp.sum(X)),
                          [cp.sum(cp.exp(cp.diag(X))) <= 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_diag_mat_maximize_entropy(self):
        """Maximize entropy of diagonal elements."""
        X = cp.Variable((3, 3), bounds=[0.1, 2])
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(cp.diag(X)))),
                          [cp.sum(cp.diag(X)) == 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        # Maximum entropy when all equal to 1
        assert np.allclose(np.diag(X.value), np.ones(3), atol=0.1)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestUpperTriIntegration:
    """Integration tests for upper_tri atom."""

    def test_upper_tri_minimize_sum(self):
        """Minimize sum of upper triangular elements with nonlinear objective."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        # Use log on the whole matrix, then take upper_tri
        prob = cp.Problem(cp.Minimize(cp.sum(cp.upper_tri(cp.log(X)))),
                          [cp.sum(X) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_upper_tri_in_linear_objective(self):
        """Use upper_tri with linear objective and nonlinear constraint."""
        X = cp.Variable((4, 4), bounds=[0.1, 5])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.upper_tri(X))),
                          [cp.sum(cp.log(X)) >= -5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestCumsumIntegration:
    """Integration tests for cumsum atom."""

    def test_cumsum_minimize_log(self):
        """Minimize sum of log of cumsum."""
        x = cp.Variable(4, bounds=[0.1, 5])
        y = cp.cumsum(x)  # y[i] = sum(x[0:i+1])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(y))),
                          [x >= 0.2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_cumsum_constraint(self):
        """Use cumsum in constraint."""
        x = cp.Variable(5, bounds=[0.1, 10])
        y = cp.cumsum(x)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [y[-1] >= 5,  # sum(x) >= 5
                           cp.sum(cp.exp(y)) <= 1000])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_cumsum_maximize_entropy(self):
        """Maximize entropy of cumsum."""
        x = cp.Variable(3, bounds=[0.1, 2])
        y = cp.cumsum(x)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(y))),
                          [y[-1] <= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_cumsum_2d_axis0(self):
        """Cumsum along axis 0 for 2D variable."""
        X = cp.Variable((3, 2), bounds=[0.1, 5])
        Y = cp.cumsum(X, axis=0)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(Y))),
                          [cp.sum(X) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestConvolveIntegration:
    """Integration tests for convolve atom."""

    def test_convolve_minimize_log(self):
        """Minimize sum of log of convolution output."""
        c = np.array([1.0, 2.0, 1.0])
        x = cp.Variable(3, bounds=[0.1, 5])
        y = convolve(c, x)  # length 5
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(y))),
                          [cp.sum(x) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_convolve_smoothing_filter(self):
        """Use convolve as a smoothing filter."""
        c = np.array([0.25, 0.5, 0.25])  # smoothing kernel
        x = cp.Variable(5, bounds=[0.1, 10])
        y = convolve(c, x)
        # Minimize variance of smoothed output
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(y - cp.sum(y)/y.size, 2))),
                          [cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_convolve_constraint(self):
        """Use convolve in constraint."""
        c = np.array([1.0, 1.0])
        x = cp.Variable(4, bounds=[0.1, 5])
        y = convolve(c, x)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [cp.sum(cp.exp(y)) <= 100])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestConcatenateIntegration:
    """Integration tests for concatenate atom."""

    def test_concatenate_minimize_log(self):
        """Minimize sum of log of concatenated variables."""
        x = cp.Variable(3, bounds=[0.1, 5])
        y = cp.Variable(2, bounds=[0.1, 5])
        z = cp.concatenate([x, y])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(z))),
                          [cp.sum(x) + cp.sum(y) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_concatenate_mixed_atoms(self):
        """Concatenate expressions with different atoms."""
        x = cp.Variable(2, bounds=[0.1, 5])
        y = cp.Variable(2, bounds=[-2, 2])
        z = cp.concatenate([cp.log(x), cp.exp(y)])
        prob = cp.Problem(cp.Minimize(cp.sum(z)),
                          [cp.sum(x) >= 2, cp.sum(y) <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_concatenate_2d_axis0(self):
        """Concatenate 2D arrays along axis 0."""
        X = cp.Variable((2, 3), bounds=[0.1, 5])
        Y = cp.Variable((2, 3), bounds=[0.1, 5])
        Z = cp.concatenate([X, Y], axis=0)  # (4, 3)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(Z))),
                          [cp.sum(X) + cp.sum(Y) >= 6])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_concatenate_2d_axis1(self):
        """Concatenate 2D arrays along axis 1."""
        X = cp.Variable((3, 2), bounds=[0.1, 5])
        Y = cp.Variable((3, 2), bounds=[0.1, 5])
        Z = cp.concatenate([X, Y], axis=1)  # (3, 4)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(Z))),
                          [cp.sum(X) + cp.sum(Y) >= 6])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_concatenate_constraint(self):
        """Use concatenate in constraint."""
        x = cp.Variable(3, bounds=[0.1, 5])
        y = cp.Variable(2, bounds=[0.1, 5])
        z = cp.concatenate([x, y])
        prob = cp.Problem(cp.Minimize(cp.sum(x) + cp.sum(y)),
                          [cp.sum(cp.exp(z)) <= 20])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestWrapIntegration:
    """Integration tests for Wrap atom (nonneg_wrap)."""

    def test_nonneg_wrap_minimize(self):
        """Minimize sum of nonneg_wrap(log(x))."""
        x = cp.Variable(3, bounds=[0.1, 5])
        prob = cp.Problem(cp.Minimize(cp.sum(nonneg_wrap(cp.log(x)))),
                          [cp.sum(x) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_nonneg_wrap_constraint(self):
        """Use nonneg_wrap in constraint."""
        x = cp.Variable(3, bounds=[0.1, 10])
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [cp.sum(nonneg_wrap(cp.log(x))) >= 0])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        # log(x) >= 0 means x >= 1
        assert np.all(x.value >= 1.0 - 1e-3)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestComposedAffineAtoms:
    """Tests for compositions of multiple affine atoms."""

    def test_trace_of_diag_vec(self):
        """trace(diag(v)) should equal sum(v)."""
        v = cp.Variable(4, bounds=[0.1, 5])
        prob = cp.Problem(cp.Minimize(cp.trace(cp.exp(cp.diag(v)))),
                          [cp.sum(v) >= 4])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_diag_of_cumsum(self):
        """Extract diagonal from matrix with cumsum entries."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        # Cumsum each column, then extract diagonal
        Y = cp.cumsum(X, axis=0)
        d = cp.diag(Y)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(d))),
                          [cp.sum(X) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_concatenate_with_upper_tri(self):
        """Concatenate upper triangular elements with a vector."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        v = cp.Variable(2, bounds=[0.1, 5])
        z = cp.concatenate([cp.reshape(cp.upper_tri(X), (3,)), v])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(z))),
                          [cp.sum(X) + cp.sum(v) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_cumsum_of_concatenate(self):
        """Cumsum of concatenated vectors."""
        x = cp.Variable(3, bounds=[0.1, 5])
        y = cp.Variable(2, bounds=[0.1, 5])
        z = cp.cumsum(cp.concatenate([x, y]))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(z))),
                          [z[-1] >= 3])  # sum(x) + sum(y) >= 3
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_convolve_with_diag(self):
        """Convolve kernel with diagonal elements."""
        X = cp.Variable((4, 4), bounds=[0.1, 5])
        c = np.array([1.0, 1.0])
        d = cp.diag(X)
        y = convolve(c, d)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(y))),
                          [cp.sum(X) >= 8])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineAtomsWithMatmul:
    """Test affine atoms combined with matrix multiplication."""

    def test_trace_with_elementwise(self):
        """Minimize exp(trace(X)) with elementwise operations."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        # Use elementwise multiply instead of matmul to avoid conj
        A = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
        prob = cp.Problem(cp.Minimize(cp.exp(cp.trace(cp.multiply(X, A)))),
                          [cp.sum(X) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_diag_of_product(self):
        """Extract diagonal of matrix product."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        d = cp.diag(X @ A)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.log(d))),
                          [cp.sum(X) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineAtomsValueVerification:
    """Verify that affine atom values are computed correctly after optimization."""

    def test_trace_value(self):
        """Verify trace value is sum of diagonal."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        expr = cp.trace(X)
        prob = cp.Problem(cp.Minimize(expr), [cp.sum(X) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(expr.value, np.trace(X.value))

    def test_diag_vec_value(self):
        """Verify diag(vector) creates correct diagonal matrix."""
        v = cp.Variable(3, bounds=[0.1, 5])
        D = cp.diag(v)
        prob = cp.Problem(cp.Minimize(cp.sum(v)), [cp.sum(v) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        expected = np.diag(v.value)
        assert np.allclose(D.value, expected)

    def test_diag_mat_value(self):
        """Verify diag(matrix) extracts diagonal correctly."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        d = cp.diag(X)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [cp.sum(X) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        expected = np.diag(X.value)
        assert np.allclose(d.value, expected)

    def test_upper_tri_value(self):
        """Verify upper_tri extracts correct elements."""
        X = cp.Variable((3, 3), bounds=[0.1, 5])
        ut = cp.upper_tri(X)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [cp.sum(X) >= 5])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        expected = X.value[np.triu_indices(3, k=1)]
        assert np.allclose(ut.value.flatten(), expected)

    def test_cumsum_value(self):
        """Verify cumsum computes cumulative sum correctly."""
        x = cp.Variable(4, bounds=[0.1, 5])
        y = cp.cumsum(x)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x) >= 4])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        expected = np.cumsum(x.value)
        assert np.allclose(y.value, expected)

    def test_convolve_value(self):
        """Verify convolve computes convolution correctly."""
        c = np.array([1.0, 2.0, 1.0])
        x = cp.Variable(3, bounds=[0.1, 5])
        y = convolve(c, x)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x) >= 3])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        expected = np.convolve(c, x.value)
        assert np.allclose(y.value, expected)

    def test_concatenate_value(self):
        """Verify concatenate combines arrays correctly."""
        x = cp.Variable(3, bounds=[0.1, 5])
        y = cp.Variable(2, bounds=[0.1, 5])
        z = cp.concatenate([x, y])
        prob = cp.Problem(cp.Minimize(cp.sum(x) + cp.sum(y)),
                          [cp.sum(x) >= 2, cp.sum(y) >= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        expected = np.concatenate([x.value, y.value])
        assert np.allclose(z.value, expected)

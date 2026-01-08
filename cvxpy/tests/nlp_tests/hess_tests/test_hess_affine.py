"""Tests for hess_vec implementations of affine atoms.

Tests for trace, diag_vec, diag_mat, upper_tri, cumsum, convolve, concatenate, and Wrap.
"""
import numpy as np

import cvxpy as cp
from cvxpy.atoms.affine.conv import convolve
from cvxpy.atoms.affine.wraps import nonneg_wrap


class TestHessTrace:
    """Tests for the hess_vec of trace atom."""

    def test_trace_hess_vec(self):
        """Test hess_vec for trace(log(X))."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.trace(cp.log(X))
        vec = np.array([1.0])  # scalar output
        result_dict = expr.hess_vec(vec)

        # trace(log(X)) = log(X[0,0]) + log(X[1,1]) + log(X[2,2])
        # Hessian w.r.t. X[i,i] is -1/X[i,i]^2 (diagonal elements only)
        rows, cols, vals = result_dict[(X, X)]
        computed_hess = np.zeros((9, 9))
        computed_hess[rows, cols] = vals

        # Expected: diagonal entries at (0,0), (4,4), (8,8) in flattened indices
        correct_hess = np.zeros((9, 9))
        correct_hess[0, 0] = -vec[0] / X.value[0, 0]**2
        correct_hess[4, 4] = -vec[0] / X.value[1, 1]**2
        correct_hess[8, 8] = -vec[0] / X.value[2, 2]**2

        assert np.allclose(computed_hess, correct_hess)


class TestHessDiagVec:
    """Tests for the hess_vec of diag_vec atom."""

    def test_diag_vec_hess_vec(self):
        """Test hess_vec for diag(exp(v))."""
        v = cp.Variable(3, name='v')
        v.value = np.array([0.0, 1.0, 2.0])

        expr = cp.diag(cp.exp(v))
        # Output shape is (3, 3) = 9 elements
        vec = np.zeros(9)
        vec[0] = 1.0  # Apply to output[0,0] = exp(v[0])
        vec[4] = 2.0  # Apply to output[1,1] = exp(v[1])
        vec[8] = 3.0  # Apply to output[2,2] = exp(v[2])

        result_dict = expr.hess_vec(vec)

        rows, cols, vals = result_dict[(v, v)]
        computed_hess = np.zeros((3, 3))
        computed_hess[rows, cols] = vals

        # d^2(exp(v[i]))/dv[i]^2 = exp(v[i])
        correct_hess = np.zeros((3, 3))
        correct_hess[0, 0] = vec[0] * np.exp(v.value[0])
        correct_hess[1, 1] = vec[4] * np.exp(v.value[1])
        correct_hess[2, 2] = vec[8] * np.exp(v.value[2])

        assert np.allclose(computed_hess, correct_hess)


class TestHessDiagMat:
    """Tests for the hess_vec of diag_mat atom."""

    def test_diag_mat_hess_vec(self):
        """Test hess_vec for diag(log(X))."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.diag(cp.log(X))
        # Output shape is (3,)
        vec = np.array([1.0, 2.0, 3.0])

        result_dict = expr.hess_vec(vec)

        rows, cols, vals = result_dict[(X, X)]
        computed_hess = np.zeros((9, 9))
        computed_hess[rows, cols] = vals

        # diag(log(X)) extracts log(X[0,0]), log(X[1,1]), log(X[2,2])
        # Hessian of log(X[i,i]) w.r.t. X[i,i] is -1/X[i,i]^2
        correct_hess = np.zeros((9, 9))
        correct_hess[0, 0] = -vec[0] / X.value[0, 0]**2
        correct_hess[4, 4] = -vec[1] / X.value[1, 1]**2
        correct_hess[8, 8] = -vec[2] / X.value[2, 2]**2

        assert np.allclose(computed_hess, correct_hess)


class TestHessUpperTri:
    """Tests for the hess_vec of upper_tri atom."""

    def test_upper_tri_hess_vec(self):
        """Test hess_vec for upper_tri(exp(X))."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.upper_tri(cp.exp(X))
        # Output shape is (3, 1) - extracts (0,1), (0,2), (1,2)
        vec = np.array([[1.0], [2.0], [3.0]])

        result_dict = expr.hess_vec(vec.ravel())

        rows, cols, vals = result_dict[(X, X)]
        computed_hess = np.zeros((9, 9))
        computed_hess[rows, cols] = vals

        # upper_tri extracts exp(X[0,1]), exp(X[0,2]), exp(X[1,2])
        # Flat indices: 3, 6, 7
        # Hessian of exp is exp itself
        correct_hess = np.zeros((9, 9))
        correct_hess[3, 3] = vec[0, 0] * np.exp(X.value[0, 1])
        correct_hess[6, 6] = vec[1, 0] * np.exp(X.value[0, 2])
        correct_hess[7, 7] = vec[2, 0] * np.exp(X.value[1, 2])

        assert np.allclose(computed_hess, correct_hess)


class TestHessCumsum:
    """Tests for the hess_vec of cumsum atom."""

    def test_cumsum_hess_vec(self):
        """Test hess_vec for cumsum(log(x))."""
        x = cp.Variable(3, name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        expr = cp.cumsum(cp.log(x))
        vec = np.array([1.0, 1.0, 1.0])

        result_dict = expr.hess_vec(vec)

        rows, cols, vals = result_dict[(x, x)]
        computed_hess = np.zeros((3, 3))
        computed_hess[rows, cols] = vals

        # The gradient matrix for cumsum is upper triangular (CVXPY convention)
        # grad_matrix.T @ vec gives the transformed vec for the child's hess_vec
        # For cumsum with upper-tri gradient, grad_matrix.T is lower triangular
        # transformed_vec = lower_tri @ [1,1,1] = [1, 2, 3]
        # Then log.hess_vec([1,2,3]) gives -[1,2,3]/x^2
        correct_hess = np.zeros((3, 3))
        transformed_vec = np.tril(np.ones((3, 3))) @ vec  # [1, 2, 3]
        correct_hess[0, 0] = -transformed_vec[0] / x.value[0]**2
        correct_hess[1, 1] = -transformed_vec[1] / x.value[1]**2
        correct_hess[2, 2] = -transformed_vec[2] / x.value[2]**2

        assert np.allclose(computed_hess, correct_hess)


class TestHessConvolve:
    """Tests for the hess_vec of convolve atom."""

    def test_convolve_hess_vec(self):
        """Test hess_vec for convolve(c, log(x))."""
        c = np.array([1.0, 2.0])
        x = cp.Variable(3, name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        expr = convolve(c, cp.log(x))
        # Output shape is (4,)
        vec = np.array([1.0, 1.0, 1.0, 1.0])

        result_dict = expr.hess_vec(vec)

        rows, cols, vals = result_dict[(x, x)]
        computed_hess = np.zeros((3, 3))
        computed_hess[rows, cols] = vals

        # convolve(c, log(x)) output:
        # y[0] = c[0]*log(x[0])
        # y[1] = c[1]*log(x[0]) + c[0]*log(x[1])
        # y[2] = c[1]*log(x[1]) + c[0]*log(x[2])
        # y[3] = c[1]*log(x[2])
        # After applying vec=[1,1,1,1]:
        # Coefficient for log(x[0]): c[0]*vec[0] + c[1]*vec[1] = 1*1 + 2*1 = 3
        # Coefficient for log(x[1]): c[0]*vec[1] + c[1]*vec[2] = 1*1 + 2*1 = 3
        # Coefficient for log(x[2]): c[0]*vec[2] + c[1]*vec[3] = 1*1 + 2*1 = 3
        # Hessian: -coeff/x^2
        correct_hess = np.zeros((3, 3))
        correct_hess[0, 0] = -3.0 / x.value[0]**2
        correct_hess[1, 1] = -3.0 / x.value[1]**2
        correct_hess[2, 2] = -3.0 / x.value[2]**2

        assert np.allclose(computed_hess, correct_hess)


class TestHessConcatenate:
    """Tests for the hess_vec of concatenate atom."""

    def test_concatenate_hess_vec(self):
        """Test hess_vec for concatenate([log(x), exp(y)])."""
        x = cp.Variable(2, name='x')
        y = cp.Variable(2, name='y')
        x.value = np.array([1.0, 2.0])
        y.value = np.array([0.0, 1.0])

        expr = cp.concatenate([cp.log(x), cp.exp(y)])
        # Output shape is (4,)
        vec = np.array([1.0, 2.0, 3.0, 4.0])

        result_dict = expr.hess_vec(vec)

        # Hessian w.r.t. (x, x)
        rows, cols, vals = result_dict[(x, x)]
        computed_hess_x = np.zeros((2, 2))
        computed_hess_x[rows, cols] = vals

        # log(x) contributes: -vec[0]/x[0]^2, -vec[1]/x[1]^2
        correct_hess_x = np.zeros((2, 2))
        correct_hess_x[0, 0] = -vec[0] / x.value[0]**2
        correct_hess_x[1, 1] = -vec[1] / x.value[1]**2

        assert np.allclose(computed_hess_x, correct_hess_x)

        # Hessian w.r.t. (y, y)
        rows, cols, vals = result_dict[(y, y)]
        computed_hess_y = np.zeros((2, 2))
        computed_hess_y[rows, cols] = vals

        # exp(y) contributes: vec[2]*exp(y[0]), vec[3]*exp(y[1])
        correct_hess_y = np.zeros((2, 2))
        correct_hess_y[0, 0] = vec[2] * np.exp(y.value[0])
        correct_hess_y[1, 1] = vec[3] * np.exp(y.value[1])

        assert np.allclose(computed_hess_y, correct_hess_y)

    def test_concatenate_hess_vec_2d(self):
        """Test hess_vec for concatenate along axis 1."""
        X = cp.Variable((2, 2), name='X')
        Y = cp.Variable((2, 2), name='Y')
        X.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y.value = np.array([[1.0, 1.0], [1.0, 1.0]])

        expr = cp.concatenate([cp.log(X), cp.exp(Y)], axis=1)
        # Output shape is (2, 4)
        vec = np.ones(8)

        result_dict = expr.hess_vec(vec)

        # Hessian w.r.t. (X, X)
        rows, cols, vals = result_dict[(X, X)]
        computed_hess_x = np.zeros((4, 4))
        computed_hess_x[rows, cols] = vals

        # log(X) contributes -1/X^2 on diagonal
        correct_hess_x = np.diag(-1.0 / X.value.ravel('F')**2)

        assert np.allclose(computed_hess_x, correct_hess_x)


class TestHessWrap:
    """Tests for the hess_vec of Wrap atom."""

    def test_nonneg_wrap_hess_vec(self):
        """Test hess_vec for nonneg_wrap(log(x))."""
        x = cp.Variable(3, name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        expr = nonneg_wrap(cp.log(x))
        vec = np.array([1.0, 2.0, 3.0])

        result_dict = expr.hess_vec(vec)

        rows, cols, vals = result_dict[(x, x)]
        computed_hess = np.zeros((3, 3))
        computed_hess[rows, cols] = vals

        # Wrap is identity, so Hessian is just d^2(log(x))/dx^2 = -vec/x^2
        correct_hess = np.diag(-vec / x.value**2)

        assert np.allclose(computed_hess, correct_hess)

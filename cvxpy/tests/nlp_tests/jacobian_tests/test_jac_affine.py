"""Tests for Jacobian implementations of affine atoms.

Tests for trace, diag_vec, diag_mat, upper_tri, cumsum, convolve, concatenate, and Wrap.
"""
import numpy as np

import cvxpy as cp
from cvxpy.atoms.affine.conv import convolve
from cvxpy.atoms.affine.wraps import nonneg_wrap


class TestJacTrace:
    """Tests for the Jacobian of trace atom."""

    def test_trace_simple(self):
        """Test trace of a matrix variable."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.trace(X)
        assert expr.shape == ()
        result_dict = expr.jacobian()

        # trace(X) = X[0,0] + X[1,1] + X[2,2]
        # In column-major order, diagonal indices are 0, 4, 8
        computed_jac = np.zeros((1, 9))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((1, 9))
        correct_jac[0, 0] = 1.0  # X[0,0]
        correct_jac[0, 4] = 1.0  # X[1,1]
        correct_jac[0, 8] = 1.0  # X[2,2]

        assert np.allclose(computed_jac, correct_jac)

    def test_trace_with_nonlinear(self):
        """Test trace composed with log."""
        X = cp.Variable((2, 2), name='X')
        X.value = np.array([[1.0, 2.0],
                           [3.0, 4.0]])

        expr = cp.trace(cp.log(X))
        result_dict = expr.jacobian()

        # trace(log(X)) = log(X[0,0]) + log(X[1,1])
        # d/dX[i,j] = 1/X[i,j] if i==j, else 0
        computed_jac = np.zeros((1, 4))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((1, 4))
        correct_jac[0, 0] = 1.0 / X.value[0, 0]  # d/dX[0,0]
        correct_jac[0, 3] = 1.0 / X.value[1, 1]  # d/dX[1,1]

        assert np.allclose(computed_jac, correct_jac)


class TestJacDiagVec:
    """Tests for the Jacobian of diag_vec atom."""

    def test_diag_vec_simple(self):
        """Test diag of a vector (creates diagonal matrix)."""
        v = cp.Variable(3, name='v')
        v.value = np.array([1.0, 2.0, 3.0])

        expr = cp.diag(v)
        assert expr.shape == (3, 3)
        result_dict = expr.jacobian()

        # diag(v) creates a 3x3 matrix with v on the diagonal
        # In column-major order, output indices for diagonal are 0, 4, 8
        computed_jac = np.zeros((9, 3))
        rows, cols, vals = result_dict[v]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((9, 3))
        correct_jac[0, 0] = 1.0  # v[0] -> output[0,0] (flat idx 0)
        correct_jac[4, 1] = 1.0  # v[1] -> output[1,1] (flat idx 4)
        correct_jac[8, 2] = 1.0  # v[2] -> output[2,2] (flat idx 8)

        assert np.allclose(computed_jac, correct_jac)

    def test_diag_vec_with_offset(self):
        """Test diag of a vector with offset k=1."""
        v = cp.Variable(2, name='v')
        v.value = np.array([1.0, 2.0])

        expr = cp.diag(v, k=1)
        assert expr.shape == (3, 3)
        result_dict = expr.jacobian()

        # diag(v, k=1) creates a 3x3 matrix with v on the superdiagonal
        # v[0] -> output[0,1] (flat idx 3), v[1] -> output[1,2] (flat idx 7)
        computed_jac = np.zeros((9, 2))
        rows, cols, vals = result_dict[v]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((9, 2))
        correct_jac[3, 0] = 1.0  # v[0] -> output[0,1]
        correct_jac[7, 1] = 1.0  # v[1] -> output[1,2]

        assert np.allclose(computed_jac, correct_jac)

    def test_diag_vec_with_negative_offset(self):
        """Test diag of a vector with offset k=-1."""
        v = cp.Variable(2, name='v')
        v.value = np.array([1.0, 2.0])

        expr = cp.diag(v, k=-1)
        assert expr.shape == (3, 3)
        result_dict = expr.jacobian()

        # diag(v, k=-1) creates a 3x3 matrix with v on the subdiagonal
        # v[0] -> output[1,0] (flat idx 1), v[1] -> output[2,1] (flat idx 5)
        computed_jac = np.zeros((9, 2))
        rows, cols, vals = result_dict[v]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((9, 2))
        correct_jac[1, 0] = 1.0  # v[0] -> output[1,0]
        correct_jac[5, 1] = 1.0  # v[1] -> output[2,1]

        assert np.allclose(computed_jac, correct_jac)

    def test_diag_vec_with_nonlinear(self):
        """Test diag composed with exp."""
        v = cp.Variable(2, name='v')
        v.value = np.array([1.0, 2.0])

        expr = cp.diag(cp.exp(v))
        result_dict = expr.jacobian()

        computed_jac = np.zeros((4, 2))
        rows, cols, vals = result_dict[v]
        computed_jac[rows, cols] = vals

        # d(diag(exp(v)))/dv[i] = exp(v[i]) at diagonal position
        correct_jac = np.zeros((4, 2))
        correct_jac[0, 0] = np.exp(v.value[0])  # v[0] -> output[0,0]
        correct_jac[3, 1] = np.exp(v.value[1])  # v[1] -> output[1,1]

        assert np.allclose(computed_jac, correct_jac)


class TestJacDiagMat:
    """Tests for the Jacobian of diag_mat atom."""

    def test_diag_mat_simple(self):
        """Test diag of a matrix (extracts diagonal)."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.diag(X)
        assert expr.shape == (3,)
        result_dict = expr.jacobian()

        # diag(X) extracts diagonal elements
        # In column-major order, input diagonal indices are 0, 4, 8
        computed_jac = np.zeros((3, 9))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((3, 9))
        correct_jac[0, 0] = 1.0  # X[0,0] -> output[0]
        correct_jac[1, 4] = 1.0  # X[1,1] -> output[1]
        correct_jac[2, 8] = 1.0  # X[2,2] -> output[2]

        assert np.allclose(computed_jac, correct_jac)

    def test_diag_mat_with_offset(self):
        """Test diag of a matrix with offset k=1."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.arange(1.0, 10.0).reshape(3, 3)

        expr = cp.diag(X, k=1)
        assert expr.shape == (2,)
        result_dict = expr.jacobian()

        # diag(X, k=1) extracts superdiagonal: X[0,1], X[1,2]
        # Flat indices: 3, 7
        computed_jac = np.zeros((2, 9))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((2, 9))
        correct_jac[0, 3] = 1.0  # X[0,1] -> output[0]
        correct_jac[1, 7] = 1.0  # X[1,2] -> output[1]

        assert np.allclose(computed_jac, correct_jac)

    def test_diag_mat_with_nonlinear(self):
        """Test diag(log(X))."""
        X = cp.Variable((2, 2), name='X')
        X.value = np.array([[1.0, 2.0],
                           [3.0, 4.0]])

        expr = cp.diag(cp.log(X))
        result_dict = expr.jacobian()

        computed_jac = np.zeros((2, 4))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        # d(diag(log(X)))/dX[i,j] = 1/X[i,j] if i==j
        correct_jac = np.zeros((2, 4))
        correct_jac[0, 0] = 1.0 / X.value[0, 0]  # X[0,0] -> output[0]
        correct_jac[1, 3] = 1.0 / X.value[1, 1]  # X[1,1] -> output[1]

        assert np.allclose(computed_jac, correct_jac)


class TestJacUpperTri:
    """Tests for the Jacobian of upper_tri atom."""

    def test_upper_tri_simple(self):
        """Test upper_tri of a matrix."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.upper_tri(X)
        assert expr.shape == (3, 1)  # n*(n-1)/2 = 3
        result_dict = expr.jacobian()

        # upper_tri extracts elements (0,1), (0,2), (1,2) in row-major order
        # Output indices: 0, 1, 2
        # Input flat indices (col-major): (0,1)->3, (0,2)->6, (1,2)->7
        computed_jac = np.zeros((3, 9))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        correct_jac = np.zeros((3, 9))
        correct_jac[0, 3] = 1.0  # X[0,1] -> output[0]
        correct_jac[1, 6] = 1.0  # X[0,2] -> output[1]
        correct_jac[2, 7] = 1.0  # X[1,2] -> output[2]

        assert np.allclose(computed_jac, correct_jac)

    def test_upper_tri_4x4(self):
        """Test upper_tri of a 4x4 matrix."""
        X = cp.Variable((4, 4), name='X')
        X.value = np.arange(1.0, 17.0).reshape(4, 4)

        expr = cp.upper_tri(X)
        assert expr.shape == (6, 1)  # 4*3/2 = 6
        result_dict = expr.jacobian()

        # Elements: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # Flat indices (col-major): 4, 8, 12, 9, 13, 14
        computed_jac = np.zeros((6, 16))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        expected_input_indices = [4, 8, 12, 9, 13, 14]
        for out_idx, in_idx in enumerate(expected_input_indices):
            assert computed_jac[out_idx, in_idx] == 1.0

    def test_upper_tri_with_nonlinear(self):
        """Test upper_tri(exp(X))."""
        X = cp.Variable((3, 3), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        expr = cp.upper_tri(cp.exp(X))
        result_dict = expr.jacobian()

        computed_jac = np.zeros((3, 9))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        # d(upper_tri(exp(X)))/dX = exp(X) at upper tri positions
        correct_jac = np.zeros((3, 9))
        correct_jac[0, 3] = np.exp(X.value[0, 1])  # X[0,1]
        correct_jac[1, 6] = np.exp(X.value[0, 2])  # X[0,2]
        correct_jac[2, 7] = np.exp(X.value[1, 2])  # X[1,2]

        assert np.allclose(computed_jac, correct_jac)


class TestJacCumsum:
    """Tests for the Jacobian of cumsum atom."""

    def test_cumsum_1d(self):
        """Test cumsum of a 1D vector."""
        x = cp.Variable(4, name='x')
        x.value = np.array([1.0, 2.0, 3.0, 4.0])

        expr = cp.cumsum(x)
        assert expr.shape == (4,)
        result_dict = expr.jacobian()

        # cumsum([a, b, c, d]) = [a, a+b, a+b+c, a+b+c+d]
        # CVXPY Jacobian convention: grad[i,j] = d(output[j])/d(input[i])
        # So the Jacobian is upper triangular of ones
        computed_jac = np.zeros((4, 4))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals

        correct_jac = np.triu(np.ones((4, 4)))

        assert np.allclose(computed_jac, correct_jac)

    def test_cumsum_with_axis0(self):
        """Test cumsum along axis 0 of a 2D matrix."""
        X = cp.Variable((3, 2), name='X')
        X.value = np.arange(1.0, 7.0).reshape(3, 2)

        expr = cp.cumsum(X, axis=0)
        assert expr.shape == (3, 2)
        result_dict = expr.jacobian()

        computed_jac = np.zeros((6, 6))
        rows, cols, vals = result_dict[X]
        computed_jac[rows, cols] = vals

        # cumsum along axis 0: each column is independently cumsumed
        # CVXPY convention: grad[i,j] = d(output[j])/d(input[i])
        # In Fortran order, column 0 is indices 0,1,2 and column 1 is 3,4,5
        correct_jac = np.zeros((6, 6))
        # Column 0: output depends on earlier inputs
        correct_jac[0, 0] = 1.0
        correct_jac[0, 1] = 1.0
        correct_jac[0, 2] = 1.0
        correct_jac[1, 1] = 1.0
        correct_jac[1, 2] = 1.0
        correct_jac[2, 2] = 1.0
        # Column 1
        correct_jac[3, 3] = 1.0
        correct_jac[3, 4] = 1.0
        correct_jac[3, 5] = 1.0
        correct_jac[4, 4] = 1.0
        correct_jac[4, 5] = 1.0
        correct_jac[5, 5] = 1.0

        assert np.allclose(computed_jac, correct_jac)

    def test_cumsum_with_nonlinear(self):
        """Test cumsum(log(x))."""
        x = cp.Variable(3, name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        expr = cp.cumsum(cp.log(x))
        result_dict = expr.jacobian()

        computed_jac = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals

        # d(cumsum(log(x)))/dx with CVXPY convention: upper_tri @ diag(1/x)
        log_grad = np.diag(1.0 / x.value)
        cumsum_grad = np.triu(np.ones((3, 3)))
        correct_jac = cumsum_grad @ log_grad

        assert np.allclose(computed_jac, correct_jac)


class TestJacConvolve:
    """Tests for the Jacobian of convolve atom."""

    def test_convolve_simple(self):
        """Test convolution with a constant kernel."""
        c = np.array([1.0, 2.0, 1.0])
        x = cp.Variable(3, name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        expr = convolve(c, x)
        assert expr.shape == (5,)  # 3 + 3 - 1
        result_dict = expr.jacobian()

        # Convolution is linear: y = Toeplitz(c) @ x
        # y[k] = sum_j c[k-j] * x[j]
        # Standard Jacobian (rows=output, cols=input):
        #       x[0]  x[1]  x[2]
        # y[0]  1     0     0
        # y[1]  2     1     0
        # y[2]  1     2     1
        # y[3]  0     1     2
        # y[4]  0     0     1
        computed_jac = np.zeros((5, 3))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals

        # Toeplitz matrix for convolution
        correct_jac = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0]
        ])

        assert np.allclose(computed_jac, correct_jac)

    def test_convolve_scalar_kernel(self):
        """Test convolution with scalar kernel."""
        c = np.array([3.0])
        x = cp.Variable(4, name='x')
        x.value = np.array([1.0, 2.0, 3.0, 4.0])

        expr = convolve(c, x)
        assert expr.shape == (4,)
        result_dict = expr.jacobian()

        computed_jac = np.zeros((4, 4))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals

        # Scalar convolution is just scaling (same either way)
        correct_jac = 3.0 * np.eye(4)

        assert np.allclose(computed_jac, correct_jac)

    def test_convolve_with_nonlinear(self):
        """Test convolve(c, exp(x))."""
        c = np.array([1.0, 1.0])
        x = cp.Variable(3, name='x')
        x.value = np.array([0.0, 1.0, 2.0])

        expr = convolve(c, cp.exp(x))
        result_dict = expr.jacobian()

        # Standard Jacobian (rows=output, cols=input)
        computed_jac = np.zeros((4, 3))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals

        # d(conv(c, exp(x)))/dx = Toeplitz(c) @ diag(exp(x))
        standard_toeplitz = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ])
        exp_diag = np.diag(np.exp(x.value))
        correct_jac = standard_toeplitz @ exp_diag

        assert np.allclose(computed_jac, correct_jac)


class TestJacConcatenate:
    """Tests for the Jacobian of concatenate atom."""

    def test_concatenate_1d(self):
        """Test concatenate of 1D vectors along axis 0."""
        x = cp.Variable(3, name='x')
        y = cp.Variable(2, name='y')
        x.value = np.array([1.0, 2.0, 3.0])
        y.value = np.array([4.0, 5.0])

        expr = cp.concatenate([x, y])
        assert expr.shape == (5,)
        result_dict = expr.jacobian()

        # x maps to output indices 0, 1, 2
        computed_jac_x = np.zeros((5, 3))
        rows, cols, vals = result_dict[x]
        computed_jac_x[rows, cols] = vals

        correct_jac_x = np.zeros((5, 3))
        correct_jac_x[0, 0] = 1.0
        correct_jac_x[1, 1] = 1.0
        correct_jac_x[2, 2] = 1.0

        assert np.allclose(computed_jac_x, correct_jac_x)

        # y maps to output indices 3, 4
        computed_jac_y = np.zeros((5, 2))
        rows, cols, vals = result_dict[y]
        computed_jac_y[rows, cols] = vals

        correct_jac_y = np.zeros((5, 2))
        correct_jac_y[3, 0] = 1.0
        correct_jac_y[4, 1] = 1.0

        assert np.allclose(computed_jac_y, correct_jac_y)

    def test_concatenate_2d_axis0(self):
        """Test concatenate of 2D matrices along axis 0."""
        X = cp.Variable((2, 3), name='X')
        Y = cp.Variable((2, 3), name='Y')
        X.value = np.arange(1.0, 7.0).reshape(2, 3)
        Y.value = np.arange(7.0, 13.0).reshape(2, 3)

        expr = cp.concatenate([X, Y], axis=0)
        assert expr.shape == (4, 3)
        result_dict = expr.jacobian()

        # X is at rows 0-1, Y is at rows 2-3
        # In Fortran order, X[r,c] at output[r,c] has flat idx r + c*4
        computed_jac_x = np.zeros((12, 6))
        rows, cols, vals = result_dict[X]
        computed_jac_x[rows, cols] = vals

        correct_jac_x = np.zeros((12, 6))
        for c in range(3):
            for r in range(2):
                in_idx = r + c * 2
                out_idx = r + c * 4
                correct_jac_x[out_idx, in_idx] = 1.0

        assert np.allclose(computed_jac_x, correct_jac_x)

    def test_concatenate_2d_axis1(self):
        """Test concatenate of 2D matrices along axis 1."""
        X = cp.Variable((2, 2), name='X')
        Y = cp.Variable((2, 3), name='Y')
        X.value = np.arange(1.0, 5.0).reshape(2, 2)
        Y.value = np.arange(5.0, 11.0).reshape(2, 3)

        expr = cp.concatenate([X, Y], axis=1)
        assert expr.shape == (2, 5)
        result_dict = expr.jacobian()

        # X is at cols 0-1, Y is at cols 2-4
        computed_jac_x = np.zeros((10, 4))
        rows, cols, vals = result_dict[X]
        computed_jac_x[rows, cols] = vals

        # X[r,c] maps to output[r,c], flat idx r + c*2
        correct_jac_x = np.zeros((10, 4))
        for c in range(2):
            for r in range(2):
                in_idx = r + c * 2
                out_idx = r + c * 2
                correct_jac_x[out_idx, in_idx] = 1.0

        assert np.allclose(computed_jac_x, correct_jac_x)

    def test_concatenate_with_nonlinear(self):
        """Test concatenate([log(x), exp(y)])."""
        x = cp.Variable(2, name='x')
        y = cp.Variable(2, name='y')
        x.value = np.array([1.0, 2.0])
        y.value = np.array([0.0, 1.0])

        expr = cp.concatenate([cp.log(x), cp.exp(y)])
        result_dict = expr.jacobian()

        computed_jac_x = np.zeros((4, 2))
        rows, cols, vals = result_dict[x]
        computed_jac_x[rows, cols] = vals

        correct_jac_x = np.zeros((4, 2))
        correct_jac_x[0, 0] = 1.0 / x.value[0]
        correct_jac_x[1, 1] = 1.0 / x.value[1]

        assert np.allclose(computed_jac_x, correct_jac_x)

        computed_jac_y = np.zeros((4, 2))
        rows, cols, vals = result_dict[y]
        computed_jac_y[rows, cols] = vals

        correct_jac_y = np.zeros((4, 2))
        correct_jac_y[2, 0] = np.exp(y.value[0])
        correct_jac_y[3, 1] = np.exp(y.value[1])

        assert np.allclose(computed_jac_y, correct_jac_y)


class TestJacWrap:
    """Tests for the Jacobian of Wrap atom."""

    def test_nonneg_wrap(self):
        """Test nonneg_wrap passes through Jacobian."""
        x = cp.Variable(3, name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        expr = nonneg_wrap(cp.log(x))
        result_dict = expr.jacobian()

        computed_jac = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals

        # Wrap is identity, so Jacobian is just d(log(x))/dx = diag(1/x)
        correct_jac = np.diag(1.0 / x.value)

        assert np.allclose(computed_jac, correct_jac)

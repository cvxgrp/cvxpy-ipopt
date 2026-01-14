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

Tests comparing canon backend vs diff engine for affine coefficient extraction.

The key insight is that for affine expressions f(x) = Ax + b:
- The Jacobian of f with respect to x is exactly A (constant)
- Evaluating f(0) gives b

So the diff engine's jacobian() and constraint_forward(zeros) should produce
the same [A | b] matrix as the canon backend's get_problem_matrix().
"""

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.coeff_extractor import CoeffExtractor

# Import diff engine - skip tests if not available
try:
    from dnlp_diff_engine import C_problem, build_variable_dict
    HAS_DIFFENGINE = True
except ImportError:
    HAS_DIFFENGINE = False


def get_coeffs_canon(expressions, problem):
    """
    Get A, b using canon backend.

    Args:
        expressions: list of CVXPY expressions
        problem: CVXPY Problem (for variable mapping)

    Returns:
        A: coefficient matrix (num_rows x num_vars)
        b: constant offset vector (num_rows,)
    """
    inverse_data = InverseData(problem)
    extractor = CoeffExtractor(inverse_data, cp.SCIPY_CANON_BACKEND)
    tensor = extractor.affine(expressions)

    constr_len = sum(e.size for e in expressions)
    var_len = inverse_data.x_length

    # Canon backend returns shape (constr_len * (var_len + 1), param_size + 1)
    # For non-param problems, param_size + 1 = 1
    # Reshape with Fortran order to get [A | b]
    Ab = tensor.toarray().reshape((constr_len, var_len + 1), order='F')

    A = Ab[:, :-1]  # All columns except last
    b = Ab[:, -1]   # Last column is constant offset
    return A, b


def get_coeffs_diffengine(expressions, problem):
    """
    Get A, b using diff engine.

    Args:
        expressions: list of CVXPY expressions
        problem: CVXPY Problem (for variable mapping)

    Returns:
        A: coefficient matrix (num_rows x num_vars)
        b: constant offset vector (num_rows,)
    """
    var_dict, n_vars = build_variable_dict(problem.variables())

    # Create a problem where each expression becomes an equality constraint.
    # The diff engine will compute Jacobians of these constraints.
    dummy_obj = cp.Constant(0)
    dummy_constraints = [expr == 0 for expr in expressions]
    dummy_problem = cp.Problem(cp.Minimize(dummy_obj), dummy_constraints)

    c_prob = C_problem(dummy_problem)
    c_prob.init_derivatives()

    # Evaluate at x=0 to get b (since f(0) = A*0 + b = b)
    x_zero = np.zeros(n_vars)
    b = c_prob.constraint_forward(x_zero)

    # Get Jacobian (equals A for affine expressions)
    A = c_prob.jacobian().toarray()

    return A, b


def compare_coeffs(expressions, problem, rtol=1e-10, atol=1e-10):
    """
    Compare coefficients from canon backend and diff engine.

    Returns True if they match, raises AssertionError otherwise.
    """
    A_canon, b_canon = get_coeffs_canon(expressions, problem)
    A_diff, b_diff = get_coeffs_diffengine(expressions, problem)

    np.testing.assert_allclose(A_canon, A_diff, rtol=rtol, atol=atol,
                               err_msg="A matrices don't match")
    np.testing.assert_allclose(b_canon, b_diff, rtol=rtol, atol=atol,
                               err_msg="b vectors don't match")
    return True


@pytest.mark.skipif(not HAS_DIFFENGINE, reason="dnlp_diff_engine not installed")
class TestCoeffExtractorDiffEngine:
    """Tests comparing canon backend vs diff engine for affine expressions."""

    def test_simple_sum(self):
        """Test: sum(x) + constant"""
        x = cp.Variable(3)
        expr = cp.sum(x) + 5
        problem = cp.Problem(cp.Minimize(expr))

        compare_coeffs([expr], problem)

    def test_simple_linear(self):
        """Test: x + y (two variables)"""
        x = cp.Variable(2)
        y = cp.Variable(2)
        expr = x + y
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_scalar_multiplication(self):
        """Test: 3 * x"""
        x = cp.Variable(3)
        expr = 3 * x
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_negation(self):
        """Test: -x"""
        x = cp.Variable(3)
        expr = -x
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_matrix_vector_mult(self):
        """Test: A @ x + b"""
        np.random.seed(42)
        n, m = 3, 4
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        expr = A @ x + b
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    @pytest.mark.skip(reason="right_matmul has a bug in diff engine - causes Bus error")
    def test_vector_matrix_mult(self):
        """Test: x @ A (row vector times matrix)"""
        np.random.seed(42)
        n, m = 3, 4
        x = cp.Variable(n)
        A = np.random.randn(n, m)
        expr = x @ A
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_indexing(self):
        """Test: x[1:3]"""
        x = cp.Variable(5)
        expr = x[1:4]
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_single_index(self):
        """Test: x[0]"""
        x = cp.Variable(3)
        expr = x[0]
        problem = cp.Problem(cp.Minimize(expr))

        compare_coeffs([expr], problem)

    def test_multiple_variables_with_constants(self):
        """Test: x + 2*y + 3"""
        x = cp.Variable(2)
        y = cp.Variable(2)
        expr = x + 2 * y + 3
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_multiple_expressions(self):
        """Test multiple expressions stacked together"""
        x = cp.Variable(2)
        y = cp.Variable()
        expr1 = cp.sum(x) + y + 5
        expr2 = x + y
        problem = cp.Problem(cp.Minimize(expr1))

        compare_coeffs([expr1, expr2], problem)

    def test_scalar_variable(self):
        """Test with scalar variable"""
        x = cp.Variable()
        expr = 2 * x + 3
        problem = cp.Problem(cp.Minimize(expr))

        compare_coeffs([expr], problem)

    def test_combined_operations(self):
        """Test: A @ x - 2*y + b"""
        np.random.seed(42)
        n = 3
        x = cp.Variable(n)
        y = cp.Variable(n)
        A = np.random.randn(n, n)
        b = np.random.randn(n)
        expr = A @ x - 2 * y + b
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_element_wise_multiply_constant(self):
        """Test: c * x (element-wise with constant vector)"""
        np.random.seed(42)
        n = 4
        x = cp.Variable(n)
        c = np.random.randn(n)
        expr = cp.multiply(c, x)
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_reshape(self):
        """Test reshape operation"""
        x = cp.Variable(6)
        expr = cp.reshape(x, (2, 3))
        # Flatten for comparison since reshape should be a no-op for data
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    def test_zero_constant(self):
        """Test expression with zero constant offset"""
        x = cp.Variable(3)
        expr = 2 * x  # No constant offset
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        compare_coeffs([expr], problem)

    @pytest.mark.skip(reason="Pure constant expressions don't involve variables for diff engine")
    def test_pure_constant(self):
        """Test pure constant expression (edge case)"""
        x = cp.Variable(2)
        const_expr = cp.Constant(np.array([1.0, 2.0, 3.0]))
        # Need a variable in the problem for proper variable mapping
        problem = cp.Problem(cp.Minimize(cp.sum(x)))

        # This tests that the constant offset is correctly extracted
        compare_coeffs([const_expr], problem)


@pytest.mark.skipif(not HAS_DIFFENGINE, reason="dnlp_diff_engine not installed")
class TestCoeffExtractorFormat:
    """Tests verifying output format compatibility."""

    def test_output_shape(self):
        """Verify the output shapes match expected format"""
        x = cp.Variable(3)
        expr = 2 * x + 1
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        A_canon, b_canon = get_coeffs_canon([expr], problem)
        A_diff, b_diff = get_coeffs_diffengine([expr], problem)

        # A should be (num_rows, num_vars)
        assert A_canon.shape == (3, 3)
        assert A_diff.shape == (3, 3)

        # b should be (num_rows,)
        assert b_canon.shape == (3,)
        assert b_diff.shape == (3,)

    def test_sparse_preservation(self):
        """Verify sparsity patterns are preserved"""
        np.random.seed(42)
        n = 5
        x = cp.Variable(n)
        # Create a sparse matrix
        A_sparse = np.zeros((n, n))
        A_sparse[0, 0] = 1.0
        A_sparse[1, 2] = 2.0
        A_sparse[2, 4] = 3.0

        expr = A_sparse @ x
        problem = cp.Problem(cp.Minimize(cp.sum(expr)))

        A_canon, _ = get_coeffs_canon([expr], problem)
        A_diff, _ = get_coeffs_diffengine([expr], problem)

        # Check sparsity pattern matches
        np.testing.assert_allclose(A_canon, A_diff)

        # Verify the sparse structure
        assert A_canon[0, 0] == 1.0
        assert A_canon[1, 2] == 2.0
        assert A_canon[2, 4] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

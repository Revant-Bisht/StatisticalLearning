import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.matrices import MyMatrixSolver

class TestDeterminant:
    """Tests for the determinant method."""

    def test_determinant_of_identity_matrix(self):
        """Determinant of identity matrix should be 1."""
        identity = np.eye(3)
        solver = MyMatrixSolver(identity)
        result = solver.determinant
        assert result == 1

    def test_determinant_of_simple_2x2_matrix(self):
        """Determinant of simple 2x2 matrix should be ad - bc."""
        matrix = np.array([[2, 4], [1, 3]])
        solver = MyMatrixSolver(matrix)
        result = solver.determinant
        assert result == 2 * 3 - 4 * 1

    def test_determinant_of_large_matrix(self):
        """Determinant of large matrix should be calculated correctly."""
        matrix = np.array([[ 0,  1,  0, -2,  1],
                    [ 1,  0,  3,  1,  1],
                    [ 1, -1,  1,  1,  1],
                    [ 2,  2,  1,  0,  1],
                    [ 3,  1,  1,  1,  2]])
        solver = MyMatrixSolver(matrix)
        result = solver.determinant
        assert result == 4

class TestGaussJordanElimination:
    """Tests for the Gauss-Jordan elimination (RREF) method."""

    def test_identity_matrix_unchanged(self):
        """Identity matrix should remain unchanged after RREF."""
        identity = np.eye(3)
        solver = MyMatrixSolver(identity)
        result = solver.gauss_jordan_matrix
        np.testing.assert_array_almost_equal(result, identity)

    def test_simple_2x2_matrix(self):
        """Test RREF on a simple 2x2 matrix."""
        matrix = np.array([[2, 4], [1, 3]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix
        expected = np.eye(2)  # Full rank 2x2 should reduce to identity
        np.testing.assert_array_almost_equal(result, expected)

    def test_singular_matrix(self):
        """Test RREF on a singular (rank-deficient) matrix."""
        # Rows are linearly dependent: row3 = row1 + row2
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix
        # Should have a row of zeros (rank 2)
        assert np.allclose(result[-1], 0), "Last row should be zeros for singular matrix"

    def test_augmented_matrix(self):
        """Test RREF on a system of equations (augmented matrix)."""
        # System: x + y = 3, 2x + 3y = 8 => solution x=1, y=2
        augmented = np.array([[1, 1, 3], [2, 3, 8]])
        solver = MyMatrixSolver(augmented)
        result = solver.gauss_jordan_matrix
        expected = np.array([[1, 0, 1], [0, 1, 2]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_matrix(self):
        """Zero matrix should remain zeros."""
        zeros = np.zeros((3, 3))
        solver = MyMatrixSolver(zeros)
        result = solver.gauss_jordan_matrix
        np.testing.assert_array_almost_equal(result, zeros)

    def test_rectangular_matrix_more_cols(self):
        """Test RREF on a matrix with more columns than rows."""
        matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix
        # First two columns should form identity for full row rank
        assert result[0, 0] == 1 and result[1, 1] == 1
        assert result[0, 1] == 0 and result[1, 0] == 0

    def test_rectangular_matrix_more_rows(self):
        """Test RREF on a matrix with more rows than columns."""
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix
        # Should reduce to identity in top 2x2, zeros in bottom row
        expected_top = np.eye(2)
        np.testing.assert_array_almost_equal(result[:2, :], expected_top)


class TestMatrixSolverInstantiation:
    """Tests for basic MatrixSolver functionality."""

    def test_matrix_stored_correctly(self):
        """Matrix should be stored as an attribute."""
        matrix = np.array([[1, 2], [3, 4]])
        solver = MyMatrixSolver(matrix)
        np.testing.assert_array_equal(solver.matrix, matrix)

    def test_accepts_integer_matrix(self):
        """Should handle integer matrices by converting to float internally."""
        matrix = np.array([[1, 2], [3, 4]], dtype=int)
        solver = MyMatrixSolver(matrix)
        assert solver.gauss_jordan_matrix is not None


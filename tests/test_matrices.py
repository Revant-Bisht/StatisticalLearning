import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.matrices import MyMatrixSolver

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

    def test_determinant_of_non_square_matrix(self):
        """Non-square matrix should raise ValueError in strict mode."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3, NOT square
        solver = MyMatrixSolver(matrix)
        assert solver.determinant is np.nan
        with pytest.raises(ValueError, match="Matrix is not square"):
            solver.calculate_determinant(strict=True)
        
    def test_determinant_of_singular_matrix(self):
        """Singular matrix (det=0) should still compute determinant (returns 0)."""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Singular (rows linearly dependent)
        solver = MyMatrixSolver(matrix)
        # Singular matrices have determinant 0, no error expected
        assert solver.determinant == 0

class TestGaussJordanElimination:
    """Tests for the Gauss-Jordan elimination (RREF) method."""
    def test_identity_matrix_unchanged(self):
        """Identity matrix should remain unchanged after RREF."""
        identity = np.eye(3)
        solver = MyMatrixSolver(identity)
        result = solver.gauss_jordan_matrix  # Already numpy array
        np.testing.assert_array_almost_equal(result, identity)

    def test_simple_2x2_matrix(self):
        """Test RREF on a simple 2x2 matrix."""
        matrix = np.array([[2, 4], [1, 3]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix  # Already numpy array
        expected = np.eye(2)  # Full rank 2x2 should reduce to identity
        np.testing.assert_array_almost_equal(result, expected)

    def test_singular_matrix(self):
        """Test RREF on a singular (rank-deficient) matrix."""
        # Rows are linearly dependent: row3 = row1 + row2
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix  # Already numpy array
        # Should have a row of zeros (rank 2)
        assert np.allclose(result[-1], 0), "Last row should be zeros for singular matrix"

    def test_augmented_matrix(self):
        """Test RREF on a system of equations (augmented matrix)."""
        # System: x + y = 3, 2x + 3y = 8 => solution x=1, y=2
        augmented = np.array([[1, 1, 3], [2, 3, 8]])
        solver = MyMatrixSolver(augmented)
        result = solver.gauss_jordan_matrix  # Already numpy array
        expected = np.array([[1, 0, 1], [0, 1, 2]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_matrix(self):
        """Zero matrix should remain zeros."""
        zeros = np.zeros((3, 3))
        solver = MyMatrixSolver(zeros)
        result = solver.gauss_jordan_matrix  # Already numpy array
        np.testing.assert_array_almost_equal(result, zeros)

    def test_rectangular_matrix_more_cols(self):
        """Test RREF on a matrix with more columns than rows."""
        matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix  # Already numpy array
        # First two columns should form identity for full row rank
        assert result[0, 0] == 1 and result[1, 1] == 1
        assert result[0, 1] == 0 and result[1, 0] == 0

    def test_rectangular_matrix_more_rows(self):
        """Test RREF on a matrix with more rows than columns."""
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.gauss_jordan_matrix  # Already numpy array
        # Should reduce to identity in top 2x2, zeros in bottom row
        expected_top = np.eye(2)
        np.testing.assert_array_almost_equal(result[:2, :], expected_top)

class TestInverse:
    """Tests for the inverse method."""
    def test_2_by_2_inverse(self):
        matrix = np.array([[1, 2], [3, 4]])
        solver = MyMatrixSolver(matrix)
        result = solver.inverse
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_5_by_5_inverse(self):
        """Test inverse of an invertible 5x5 matrix."""
        # Use a known invertible matrix (random but full rank)
        matrix = np.array([
            [2, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 2]
        ], dtype=float)
        solver = MyMatrixSolver(matrix)
        result = solver.inverse
        # Verify A @ A_inv = I
        product = matrix @ result
        np.testing.assert_array_almost_equal(product, np.eye(5))

    def test_singular_matrix_returns_none(self):
        """Singular matrix should return None for inverse (non-strict mode)."""
        matrix = np.array([[1, 2], [2, 4]])  # det = 0
        solver = MyMatrixSolver(matrix)
        assert solver.inverse is None

    def test_non_square_matrix_raises_strict(self):
        """Non-square matrix should raise ValueError in strict mode."""
        matrix = np.array([[2, 6, 8], [5, 3, 1]])  # Fixed: 2x3 matrix
        solver = MyMatrixSolver(matrix)
        assert solver.inverse is None
        with pytest.raises(ValueError, match="Matrix is not square"):
            solver.calculate_inverse(strict=True)

class TestRank:
    """Tests for the rank method."""
    def test_rank_of_2_by_2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        solver = MyMatrixSolver(matrix)
        result = solver.rank
        assert result == 2

    def test_rank_of_5_by_5_full_rank_matrix(self):
        """Full rank matrix should have rank equal to its dimension."""
        matrix = np.array([
            [2, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 2]
        ], dtype=float)
        solver = MyMatrixSolver(matrix)
        result = solver.rank
        assert result == 5

    def test_rank_of_singular_matrix(self):
        """Singular matrix should have rank less than its dimension."""
        matrix = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ])
        solver = MyMatrixSolver(matrix)
        result = solver.rank
        assert result == 2   

    def test_rank_of_non_square_matrix(self):
        """Non-square matrices can still have rank computed."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        solver = MyMatrixSolver(matrix)
        result = solver.rank
        assert result == 2  # Full row rank

class TestTrace:
    """Tests for the trace method."""
    def test_trace_of_2_by_2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        solver = MyMatrixSolver(matrix)
        result = solver.trace
        assert result == 5

    def test_trace_of_5_by_5_matrix(self):
        matrix = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ])
        solver = MyMatrixSolver(matrix)
        result = solver.trace
        assert result == 1 + 7 + 13 + 19 + 25   

    def test_trace_of_non_square_matrix_returns_none(self):
        """Non-square matrix should return None for trace (non-strict mode)."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        solver = MyMatrixSolver(matrix)
        assert solver.trace is None

    def test_trace_of_non_square_matrix_raises_strict(self):
        """Non-square matrix should raise ValueError in strict mode."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        solver = MyMatrixSolver(matrix)
        with pytest.raises(ValueError, match="Matrix is not square"):
            solver.calculate_trace(strict=True)

class TestCovarianceMatrix:
    """Tests for the covariance matrix method."""
    def test_covariance_matrix_of_2_by_2_matrix(self):
        """Covariance matrix of 2x2 matrix should be calculated correctly."""
        matrix = np.array([[1, 2], [3, 4]])
        solver = MyMatrixSolver(matrix)
        result = solver.covariance_matrix
        expected = np.array([[2., 2.], [2., 2.]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_covariance_matrix_of_5_by_5_matrix(self):
        """Covariance matrix of 5x5 matrix should be calculated correctly."""
        matrix = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ])
        solver = MyMatrixSolver(matrix)
        result = solver.covariance_matrix
        expected = np.array([
            [62.5, 62.5, 62.5, 62.5, 62.5],
            [62.5, 62.5, 62.5, 62.5, 62.5],
            [62.5, 62.5, 62.5, 62.5, 62.5],
            [62.5, 62.5, 62.5, 62.5, 62.5],
            [62.5, 62.5, 62.5, 62.5, 62.5]
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_covariance_matrix_of_insufficient_samples(self):
        """Covariance matrix of insufficient samples should raise ValueError in strict mode."""
        matrix = np.array([[1, 2]])
        solver = MyMatrixSolver(matrix)
        # Check it returns zeros matrix (not using 'is' - use array comparison)
        np.testing.assert_array_equal(solver.covariance_matrix, np.zeros((2, 2)))
        with pytest.raises(ValueError, match="Matrix has less than 2 samples"):
            solver.calculate_covariance_matrix(strict=True)

class TestCorrelationMatrix:
    """Tests for the correlation matrix method."""
    def test_correlation_matrix_for_1d_array_raises(self):
        """1D array should raise ValueError (solver requires 2D matrices)."""
        matrix = np.array([1, 2, 3])  # 1D array, not a matrix
        with pytest.raises(ValueError):
            MyMatrixSolver(matrix)

    def test_correlation_matrix_for_single_sample(self):
        """Matrix with only 1 sample should return zeros (insufficient for correlation)."""
        matrix = np.array([[1, 2, 3]])  # 2D but only 1 row (1 sample)
        solver = MyMatrixSolver(matrix)
        # With only 1 sample, correlation is undefined - check it handles gracefully
        np.testing.assert_array_equal(solver.correlation_matrix, np.zeros((3, 3)))

    def test_correlation_matrix_of_2_by_2_matrix(self):
        """Correlation matrix of 2x2 matrix should be calculated correctly."""
        matrix = np.array([[1, 2], [3, 4]])
        solver = MyMatrixSolver(matrix)
        # Check it returns the correct correlation matrix (use array comparison)
        result = solver.correlation_matrix
        expected = np.array([[1., 1.], [1., 1.]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_correlation_matrix_of_5_by_5_matrix(self):
        """Correlation matrix of 5x5 matrix should be calculated correctly."""
        matrix = np.array([
            [1,  2,  3,  4,  5,  6],     # base signal
            [2,  4,  6,  8, 10, 12],     # perfectly positively correlated with row 0
            [6,  5,  4,  3,  2,  1],     # perfectly negatively correlated with row 0
            [1,  3,  2,  5,  4,  6],     # moderately positively correlated
            [6,  4,  5,  2,  3,  1],     # moderately negatively correlated
        ])
        solver = MyMatrixSolver(matrix)
        result = solver.correlation_matrix
        expected = np.array([
            [1, 0.79626712,  0.42759306, -0.64607866, -0.55199703, -0.74723998],
            [0.79626712, 1, 0.5547002, -0.1142909, -0.16896382, -0.31807321],
            [0.42759306, 0.5547002, 1, 0.27472113, 0.50767308, 0.24326682],
            [-0.64607866, -0.1142909, 0.27472113, 1, 0.92049224, 0.96904275],
            [-0.55199703, -0.16896382, 0.50767308, 0.92049224, 1, 0.95624297],
            [-0.74723998, -0.31807321, 0.24326682, 0.96904275, 0.95624297, 1]
        ])
        np.testing.assert_array_almost_equal(result, expected)

class TestEigenvalues:
    """Tests for the eigenvalues method."""
    def test_eigenvalues_of_2_by_2_matrix(self):
        matrix = np.array([[2, 1], [1, 2]])
        solver = MyMatrixSolver(matrix)
        result = solver.eigenvalues
        expected = np.array([3.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

class TestNullSpace:
    """Tests for the null space method."""
    def test_null_space_of_full_rank_matrix(self):
        """Full rank matrix should have no null space."""
        matrix = np.array([[1, 2],[3, 4]])
        solver = MyMatrixSolver(matrix)
        result = solver.null_space
        assert result is None

    def test_null_space_of_singular_matrix(self):
        """Singular matrix should have a null space."""
        matrix = np.array([[1, 0, -2, 6], [-3, 6, 6, -6], [2, -3, -4, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.null_space
        expected = np.array([[ 2., -0.,  1.,  0.], [-6., -2.,  0.,  1.]])
        np.testing.assert_array_almost_equal(result, expected)

class TestEigenvectors:
    """Tests for the eigenvectors method."""
    def test_eigenvectors_of_2_by_2_matrix(self):
        matrix = np.array([[4, 1], [2, 3]])
        solver = MyMatrixSolver(matrix)
        result = solver.eigenvectors
        # Eigenvectors are returned in the same order as eigenvalues.
        # For this matrix, eigenvalues are [5, 2] so eigenvectors are ordered accordingly.
        expected = np.array([[0.70710678, 0.70710678], [-0.4472136, 0.89442719]])
        np.testing.assert_array_almost_equal(result, expected)

class TestColumnSpace:
    """Tests for the column space method."""
    def test_column_space_of_3_by_4_matrix(self):
        matrix = np.array([[1, 0, -2, 6], [-3, 6, 6, -6], [2, -3, -4, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.column_space
        expected = np.array([[ 1, -3,  2],
       [ 0,  6, -3]])
        np.testing.assert_array_almost_equal(result, expected)

class TestRowSpace:
    """Tests for the row space method."""
    def test_row_space_of_3_by_4_matrix(self):
        matrix = np.array([[1, 0, -2, 6], [-3, 6, 6, -6], [2, -3, -4, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.row_space
        expected = np.array([[1, 0, -2, 6], [0, 1, 0, 2]])
        np.testing.assert_array_almost_equal(result, expected)

class TestBasis:
    """Tests for the basis method."""
    def test_basis_of_3_by_4_matrix(self):
        matrix = np.array([[1, 0, -2, 6], [-3, 6, 6, -6], [2, -3, -4, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.basis
        expected = np.array([[1, -3, 2], [0, 6, -3]])
        np.testing.assert_array_almost_equal(result, expected)

class TestOrthonormalBasis:
    """Tests for the orthonormal basis method."""
    def test_orthonormal_basis_of_3_by_4_matrix(self):
        matrix = np.array([[1, 0, -2, 6], [-3, 6, 6, -6], [2, -3, -4, 6]])
        solver = MyMatrixSolver(matrix)
        result = solver.orthonormal_basis
        expected = np.array([[ 0.267261, -0.801784,  0.534522],
       [ 0.872872,  0.436436,  0.218218]])
        np.testing.assert_array_almost_equal(result, expected)
    def test_orthonormal_basis_of_zero_matrix(self):
        """Zero matrix has no pivot columns, so basis is empty."""
        matrix = np.zeros((3, 3))
        solver = MyMatrixSolver(matrix)
        result = solver.orthonormal_basis
        # Empty basis should return empty array
        assert result.size == 0

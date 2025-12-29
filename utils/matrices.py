from dataclasses import dataclass
from math import pi
from pyexpat import features
import numpy as np
from utils.base import BaseMatrixSolver
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@dataclass
class GaussJordanResult:
    reduced_matrix: np.ndarray
    inverse_matrix: np.ndarray

class MyMatrixSolver(BaseMatrixSolver):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.gauss_jordan_matrix = self.gauss_jordan_elimination().reduced_matrix
        self.determinant = self.calculate_determinant()
        self.inverse = self.calculate_inverse()
        self.rank = self.calculate_rank()
        self.trace = self.calculate_trace()
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.correlation_matrix = self.calculate_correlation_matrix()
        self.eigenvalues = self.calculate_eigenvalues()
        self.eigenvectors = self.calculate_eigenvectors()
        self.null_space = self.calculate_null_space()
        self.column_space = self.calculate_column_space()
        self.row_space = self.calculate_row_space()
        self.basis = self.calculate_basis()
        self.orthonormal_basis = self.calculate_orthonormal_basis()
    def summary(self):
        report = f" Matrix has \
            RREF: {self.gauss_jordan_matrix} \
            Dimensions: {self.matrix.shape} \
            Rank: {self.rank} \
            Trace: {self.trace} \
            Eigenvalues: {self.eigenvalues} \
            Eigenvectors: {self.eigenvectors} \
            Null Space: {self.null_space} \
            Column Space: {self.column_space} \
            Row Space: {self.row_space} \
            Basis: {self.basis} \
            Orthonormal Basis: {self.orthonormal_basis} \
            Covariance Matrix: {self.covariance_matrix} \
            Correlation Matrix: {self.calculate_correlation_matrix()} \
        "
        print(report)

    def gauss_jordan_elimination(self, matrix: np.ndarray = None) -> GaussJordanResult:
        """
        Function pseudocode. Example matrix:
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

        1. Copy create matrix copy and change type to float (better precision).
        2. Get matrix dimensions (rows, columns()
        3. Initialize the pivot row to 0 (first row) (row = 0).
            This will be updated (over iterations).
        4. Iterate over each column.
        5. If the pivot row is greater than or equal to the number of rows, break the loop
            I.e., don't let "row" go beyond the last row.
        6. For all row elements in the current column, find their absolute value.
            The row with the largest absolute value is the pivot row. Find the index using np.argmax().
            Add the current row index to the pivot row index to get actual row index to swap with.

        7. Check if element is zero. If so, continue (i.e., skip it).
        8. Check if pivot row index is != the current row index.
            If so, swap the rows using fancy indexing.
                A[pivot_row], A[current_row] = A[current_row], A[pivot_row]
        9. Normalize the pivot row.
            Divide the pivot row by the pivot element.
        10. Iterate over range(rows), and only IF your iterable index is != the pivot row index, do the following:
            A. For each row element in the current column, calculate the factor by which to multiply the pivot row.
                Factor = the entry we want to zero out.
            B. If abs(factor) > 0:
                Current row element = Current row element - factor * pivot row element.

        11. Increment the pivot row index by 1.
        12. Return the matrix in reduced row echelon form.
        """
        # If a specific matrix is provided (e.g., for Null Space), use it.
        # Otherwise, default to the instance's main matrix.
        if matrix is None:
            target_matrix = self.matrix
        else:
            target_matrix = matrix

        A = target_matrix.astype(float).copy()

        rows, columns = A.shape
        identity_matrix = np.eye(rows)
        TOL = 1e-10
        pivot_row = 0

        for col in range(columns):
            if pivot_row >= rows:
                break

            # Select pivot row with largest absolute value in the current column (and check if it is zero)
            max_val_idx = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
            
            if abs(A[max_val_idx, col]) < TOL:
                continue
            
            # Swap rows if pivot_row index != current row index - for matrix and identity matrix
            if max_val_idx != pivot_row:
                A[[pivot_row, max_val_idx]] = A[[max_val_idx, pivot_row]]
                identity_matrix[[pivot_row, max_val_idx]] = identity_matrix[[max_val_idx, pivot_row]]

            # Normalize the pivot row for matrix and identity matrix
            pivot = A[pivot_row, col]
            A[pivot_row] /= pivot
            identity_matrix[pivot_row] /= pivot

            for i in range(rows):
                if i != pivot_row:
                    factor = A[i, col]
                    if abs(factor) > 0:
                        A[i] = A[i] - factor * A[pivot_row]
                        identity_matrix[i] = identity_matrix[i] - factor * identity_matrix[pivot_row]
            pivot_row += 1
        
        A[np.abs(A) < TOL] = 0.0

        return GaussJordanResult(reduced_matrix=A, inverse_matrix=identity_matrix)


    def _truncate_matrix(self, matrix: np.ndarray, column: int, row: int = 0):
        """
        Function pseudocode. Example matrix:
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

        For input (row, column) = (1, 1), the output should be:
        [[4, 6],
         [7, 9]]

        1. Horizontal split along column index = column. Return (left, right) array.
        2. Remove column_1 from right array.
        3. Combine the left and right arrays horizontally.
        4. Return the bottom from np.vsplit.
        """
        column_split_index = column  
        row_split_index = row  

        sub_matrix = np.hstack((matrix[:, :column_split_index], matrix[:, column_split_index + 1:]))
        result_matrix = np.vstack((sub_matrix[:row_split_index], sub_matrix[row_split_index + 1:]))

        return result_matrix

    def calculate_determinant(self, matrix: np.ndarray = None, row: int = 0, strict: bool = False):
        """
        Calculate the determinant using cofactor expansion along the first row.
        """
        if matrix is None:
            matrix = self.matrix

        if matrix.shape[0] != matrix.shape[1]:
            if strict:
                raise ValueError("Matrix is not square")
            else:
                logger.warning("Matrix is not square. Returning NaN.")
                return np.nan

        # Base case: 1x1 matrix
        if matrix.shape[0] == 1:
            return matrix[0, 0]

        # Base case: 2x2 matrix
        if matrix.shape[0] == 2:
            return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

        # Recursive case: cofactor expansion
        determinant = 0
        _, columns = matrix.shape
    
        for col in range(columns):
            multiplier = (-1) ** (row + col)
            element = matrix[row, col]
            sub_matrix = self._truncate_matrix(matrix, col, row)
            sub_determinant = self.calculate_determinant(sub_matrix, 0)
            determinant += multiplier * element * sub_determinant
            
        return determinant

    def calculate_inverse(self, strict: bool = False):
        # check if matrix is square and determinant is not 0
        if self.matrix.shape[0] != self.matrix.shape[1]:
            if strict:
                raise ValueError("Matrix is not square")
            else:
                logger.warning("Matrix is not square. Cannot compute inverse.")
                return None
        if self.determinant == 0 or np.isnan(self.determinant):
            if strict:
                raise ValueError("Matrix is singular")
            else:
                logger.warning("Matrix is singular. Cannot compute inverse.")
                return None

        # Call gauss_jordan_elimination but with return_identity = True (to solve A | I)
        # Inverse is the right half of the augmented matrix (I | A^-1)
        inverse_matrix = self.gauss_jordan_elimination().inverse_matrix
        return inverse_matrix

    def calculate_rank(self):
        """
        Calculate the rank of the matrix by counting the number of non-zero rows in the reduced row echelon form.
        """
        # Use RREF calculated in gauss_jordan_elimination().
        RREF = self.gauss_jordan_elimination().reduced_matrix
        TOL = 1e-10 # Tolerance for zero values

        # A row is "non-zero" there are any (at least one) elements that are greater than the tolerance
        non_zero_rows = np.any(np.abs(RREF) > TOL, axis=1)
        rank = np.sum(non_zero_rows)
        return rank

    def calculate_trace(self, strict: bool = False):
        """
        Calculate the trace of the matrix: the sum of the elements on the main diagonal.
        """
        rows, columns = self.matrix.shape
        if rows != columns:
            if strict:
                raise ValueError("Matrix is not square")
            else:
                logger.warning("Matrix is not square. Cannot compute trace.")
                return None

        diagonal_elements = np.diag(self.matrix)
        trace = np.sum(diagonal_elements)
        return trace

    def calculate_covariance_matrix(self, strict: bool = False):
        """
        Calculate the covariance matrix of the matrix.
        """
        
        input_matrix = self.matrix.astype(float).copy()  # Copy + Better precision
        n_samples, n_features = input_matrix.shape

        sufficient_samples = (n_samples >= 2)
        
        if sufficient_samples:
            # Calculate mean of each feature/ column to get mean vector
            mean_vector = np.mean(input_matrix, axis=0)
            # Centre data via broadcasting (no explicit tiling needed)
            # Subtract mean vector from input matrix to get centered matrix
            centered_matrix = input_matrix - mean_vector
            # Calculate covariance matrix using the centered matrix --> (centered_matrix^T * centered_matrix) / (n_samples - 1)
            covariance_matrix = (centered_matrix.T @ centered_matrix) / (n_samples - 1)

            return covariance_matrix
        else:
            # Insufficient samples - cannot compute covariance matrix; raise error in strict mode, return None in non-strict mode.
            error_message = f"Matrix has less than 2 samples ({n_samples} samples(s)). Cannot compute covariance matrix."
            if strict:
                raise ValueError(error_message)
            else:
                logger.warning(error_message)
                return np.zeros((n_features, n_features)) # Return zero matrix instead of NaN matrix.


    def calculate_correlation_matrix(self):
        """
        Calculate the correlation matrix of the input matrix (using covariance matrix).
        """
        covariance_matrix = self.calculate_covariance_matrix()

        # Check for empty/invalid covariance matrix results
        if covariance_matrix is None or covariance_matrix.size == 0:
            logger.warning("Covariance matrix is empty or invalid. Cannot compute correlation matrix.")
            return None
        
        # Extract diagonal elements (variances) of covariance matrix and calculate standard deviations.
        variances = np.diag(covariance_matrix)
        standard_deviations = np.sqrt(variances)
        
        # Create inverse diagonal matrix (D^-1) of standard deviations (1 / sigma).
        # Edge case - if sigma is zero, leave it as zero to avoid division by zero.
        n_features = len(standard_deviations)
        inverse_std_devs = np.zeros(n_features) # Initialize a zero matrix for our results
        TOL = 1e-10 # Tolerance for zero values

        for i in range(n_features):
            sigma = standard_deviations[i] # Get the standard deviation for the current feature
            # Check if sigma is effectively zero (constant column)
            if sigma > TOL:
                inverse_std_devs[i] = 1.0 / sigma # Calculate the inverse of the std.
            else:
                inverse_std_devs[i] = 0.0 # Leave inverse as 0 to avoid division by zero
        
        # Get diagonal matrix of inverse standard deviations.
        D_inverse = np.diag(inverse_std_devs)

        # Calculate correlation matrix: D^-1 * covariance_matrix * D^-1
        correlation_matrix = D_inverse @ covariance_matrix @ D_inverse
        return correlation_matrix

    def calculate_eigenvalues(self, max_iterations: int = 1000, tolerance: float = 1e-10, strict: bool = False):
        """
        Calculate the eigenvalues of the matrix using the QR algorithm.
        Logic:
        1. Iteratively decompose A = Q * R, then update A_new = R * Q.
            Works due to Similarity Transformation: A_{k+1} = Q^T * A_k * Q = R * Q * Q^T = R * Q.
            Therefore, A_{k+1} has the same eigenvalues as A_k (Similarity Invariance)

        2. As iterations -> infinity, A_new converges to an Upper Triangular matrix.
           The eigenvalues are the diagonal entries.
        """
        # Check for square matrix
        if self.matrix.shape[0] != self.matrix.shape[1]:
            if strict:
                raise ValueError("Matrix is not square")
            else:
                logger.warning("Matrix is not square. Cannot compute eigenvalues.")
                return None
        
        # Create a copy of the matrix and convert to float for better precision
        A_k = self.matrix.astype(float).copy()
        
        # QR Algorithm Loop
        for i in range(max_iterations):
            Q, R = np.linalg.qr(A_k)
            A_new = R @ Q   # A_{k+1} = R * Q (Similarity Transformation)
        
            A_k = A_new

        eigenvalues = np.diag(A_k)
        return eigenvalues

    def calculate_null_space(self, matrix: np.ndarray = None):
        """
        Calculate the null space a matrix using the eigenvalues.
        """
        # If a specific matrix is provided (e.g., for Null Space), use it.
        # Otherwise, default to the instance's main matrix.
        if matrix is None:
            target_matrix = self.matrix.astype(float).copy()
        else:
            target_matrix = matrix.astype(float).copy()

        # Call gauss_jordan_elimination (to use on the target matrix) to obtain RREF of the matrix. Select the reduced matrix.
        reduced_matrix = self.gauss_jordan_elimination(target_matrix).reduced_matrix

        TOL = 1e-10 # Tolerance for zero values
        rows, columns = reduced_matrix.shape

        # Identify pivot columns and rows in the reduced matrix (where leading entry = 1).
        # Intialize lists to store the pivot columns and rows.
        pivot_columns = [] 
        pivot_rows = []

        # Iterate over each row, checking all column entries for leading entry.
        for row in range(rows):
            # Get current row as vector to scan.
            current_row = reduced_matrix[row, :]

            # Find indices of all non-zero entries in the row.
            # First, obtain boolean array (True/False) for each element in the row.
            # Then use np.where() to get the indices of the True values (i.e., column indices a 1D array/list).

            non_zero_indices = np.where(np.abs(current_row) > TOL)[0] 
            if non_zero_indices.size > 0: # check that there are indeed non-zero entries in the row.
                pivot_column_index = non_zero_indices[0] # Get the first non-zero index from the list.
                pivot_columns.append(pivot_column_index) # Add the first non-zero column index to pivot_columns list.
                pivot_rows.append(row) # Add the row index to pivot_rows list.

        # Iterate over each column, checking if there is a pivot in the column (no pivot = free variable)
        free_columns = [] # Initialize list to store free columns.
        for col in range(columns):
            # Check if 'col' is not in pivot_columns list. Ignore if it is, otherwise append to free_columns list.
            is_free = True
            for pivot_column in pivot_columns:
                if col == pivot_column:
                    is_free = False
                    break
            if is_free:
                free_columns.append(col) # Add column index to free_columns list.
            
        if len(free_columns) == 0: # No free columns found - matrix is full rank.
            return None

        # Create basis for the null space by selecting the free columns.
        # Intialize list to store the basis.
        basis = []


        for free_column in free_columns:
            vector = np.zeros(columns) # creates a (row) vector of zeros with the same number of columns as the matrix.
            vector[free_column] = 1.0 # Set the free column index to 1.

            # Solve for pivot variables in terms of free variables.
            # First map which pivot row corresponds to which pivot column.
            for i in range(len(pivot_rows)):
                pivot_row = pivot_rows[i]
                pivot_column = pivot_columns[i]

                # Get the value associated with the free variable in this row.
                coefficient = reduced_matrix[pivot_row, free_column]
                # Algebra: x_pivot + (coeff * 1) = 0  =>  x_pivot = -coeff
                # Update the vector at the PIVOT COLUMN index (p_col), not p_row
                vector[pivot_column] = -coefficient

            basis.append(vector)

        # Convert basis to numpy array.
        basis = np.array(basis)
        return basis
    
    def calculate_eigenvectors(self):
        """
        Calculate the eigenvectors of the matrix.
        
        For each eigenvalue λ, finds vectors v such that Av = λv
        by computing the null space of (A - λI).
        
        Note: Uses a relaxed tolerance for null space computation because
        the QR algorithm produces approximate eigenvalues. With exact eigenvalues,
        (A - λI) would be exactly singular, but with approximations, we need
        a more lenient tolerance to detect near-singularity.
        
        Returns:
            np.ndarray: Eigenvectors as rows, ordered by DESCENDING eigenvalue magnitude.
                        This matches the convention used in PCA (largest variance first).
        """
        # Get eigenvalues using self.eigenvalues. If attribute does not exist, calculate eigenvalues.
        if self.eigenvalues is None:
            eigenvalues = self.calculate_eigenvalues()
        else:
            eigenvalues = self.eigenvalues
        
        # If eigenvalues is still None (e.g., non-square matrix), return None.
        if eigenvalues is None:
            return None
        
        rows, columns = self.matrix.shape # Get dimensions of matrix.
        identity_matrix = np.eye(rows) # Create identity matrix of same size as matrix.
        
        # Use adaptive tolerance based on matrix scale.
        # For matrices with larger values, we need a proportionally larger tolerance.
        matrix_norm = np.linalg.norm(self.matrix, ord='fro')  # Frobenius norm
        adaptive_tol = max(1e-4, matrix_norm * 1e-6)  # At least 1e-4, or scaled by matrix size
        
        TOL = 1e-10 # Tolerance for zero norm check
        eigenpairs = []  # Store (eigenvalue, eigenvector) pairs for sorting

        # Process eigenvalues in descending order for consistent output
        # Sort eigenvalues by magnitude (descending) and track which we've processed
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]  # Descending by magnitude
        processed_eigenvalues = set()

        for idx in sorted_indices:
            lambda_value = eigenvalues[idx]
            
            # Skip if we've already processed this eigenvalue (within tolerance)
            already_processed = False
            for processed in processed_eigenvalues:
                if abs(lambda_value - processed) < adaptive_tol:
                    already_processed = True
                    break
            
            if already_processed:
                continue
            
            processed_eigenvalues.add(lambda_value)
            
            # Get shifted matrix that we want the null space of (A - lambda * I)
            shifted_matrix = self.matrix - lambda_value * identity_matrix
            
            # Calculate null space with ADAPTIVE tolerance for eigenvector computation.
            # The QR algorithm gives approximate eigenvalues, so (A - λI) won't be 
            # exactly singular. Using an adaptive tolerance allows us to detect
            # the near-zero pivots that indicate null space directions.
            null_space = self._calculate_null_space_relaxed(shifted_matrix, tolerance=adaptive_tol)

            # Store and normalize the null space vectors.
            if null_space is not None:
                for vector in null_space:
                    norm = np.linalg.norm(vector)
                    if norm > TOL:
                        eigenvector = vector / norm
                        eigenpairs.append((lambda_value, eigenvector))
        
        if len(eigenpairs) == 0:
            return None
        
        # Sort by eigenvalue magnitude (descending) and extract just the eigenvectors
        eigenpairs.sort(key=lambda x: abs(x[0]), reverse=True)
        eigenvectors = np.array([pair[1] for pair in eigenpairs])
        
        return eigenvectors
    
    def _calculate_null_space_relaxed(self, matrix: np.ndarray, tolerance: float = 1e-6):
        """
        Calculate null space with adjustable tolerance for eigenvector computation.
        
        This is a variant of calculate_null_space that uses a more lenient tolerance,
        necessary when working with approximate eigenvalues from the QR algorithm.
        
        Args:
            matrix: The matrix to find the null space of.
            tolerance: Threshold for considering a value as zero (default: 1e-6).
            
        Returns:
            np.ndarray: Basis vectors for the null space (as rows), or None if full rank.
        """
        A = matrix.astype(float).copy()
        rows, columns = A.shape
        pivot_row = 0

        # Gauss-Jordan elimination with relaxed tolerance
        for col in range(columns):
            if pivot_row >= rows:
                break

            # Find pivot with largest absolute value
            max_val_idx = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row

            # Use relaxed tolerance to detect near-zero pivots
            if abs(A[max_val_idx, col]) < tolerance:
                continue

            # Swap rows if needed
            if max_val_idx != pivot_row:
                A[[pivot_row, max_val_idx]] = A[[max_val_idx, pivot_row]]

            # Normalize pivot row
            pivot = A[pivot_row, col]
            A[pivot_row] /= pivot

            # Eliminate other rows
            for i in range(rows):
                if i != pivot_row:
                    factor = A[i, col]
                    if abs(factor) > 0:
                        A[i] = A[i] - factor * A[pivot_row]
            pivot_row += 1

        # Clean up near-zero values
        A[np.abs(A) < tolerance] = 0.0

        # Identify pivot columns
        pivot_columns = []
        pivot_rows = []

        for row in range(rows):
            current_row = A[row, :]
            non_zero_indices = np.where(np.abs(current_row) > tolerance)[0]
            if non_zero_indices.size > 0:
                pivot_columns.append(non_zero_indices[0])
                pivot_rows.append(row)

        # Find free columns (columns without pivots)
        free_columns = [col for col in range(columns) if col not in pivot_columns]

        if len(free_columns) == 0:
            return None  # Matrix is full rank, no null space

        # Build basis vectors for null space
        basis = []
        for free_column in free_columns:
            vector = np.zeros(columns)
            vector[free_column] = 1.0

            # Solve for pivot variables in terms of free variable
            for i in range(len(pivot_rows)):
                p_row = pivot_rows[i]
                p_col = pivot_columns[i]
                coefficient = A[p_row, free_column]
                vector[p_col] = -coefficient

            basis.append(vector)

        return np.array(basis)

    # SECTION 2

    def calculate_column_space(self):
        """
        Calculate the column space (image) of the matrix.
        """
        RREF = self.gauss_jordan_elimination().reduced_matrix
        rows, columns = RREF.shape # Get dimensions of RREF.
        TOL = 1e-10 # Tolerance for zero values

        # Identify pivot column indices in the RREF.
        # same logic as in calculate_null_space().
        pivot_indices = []
        for row in range(rows):
            current_row = RREF[row, :]
            non_zero_indices = np.where(np.abs(current_row) > TOL)[0]
            if non_zero_indices.size > 0:
                pivot_indices.append(non_zero_indices[0])
        
        # Extract the pivot columns from the original self.matrix that correspond to pivot_columns indices.
        column_space = self.matrix[:, pivot_indices] # E.g., self.matrix[:, [0, 2]] gets columns 0 and 2.
        # Need to turn these columns into rows to iterate over them as vectors. To be used in other functions (e.g., calculate_orthonormal_basis()).
        return column_space.T


    def calculate_row_space(self):
        """
        Calculate the row space of the matrix.
        """
        RREF = self.gauss_jordan_elimination().reduced_matrix # Get RREF of the matrix.
        rows, columns = RREF.shape # Get dimensions of RREF.
        TOL = 1e-10 # Tolerance for zero values

        row_space_vectors = [] # Initialize list to store row space vectors.
        for row in range(rows):
            current_row = RREF[row, :] # Get current row as vector to scan.
            # For each row, check if there are any non-zero entries.
            if np.any(np.abs(current_row) > TOL):
                row_space_vectors.append(current_row) # If True, add the row to row_space_vectors list.
        
        return np.array(row_space_vectors) # Convert row_space_vectors list to numpy array.

    def calculate_basis(self):
        """
        Calculate basis for the column space of the matrix.
        """
        # Avoid using this function due to ambiguity in the definition of basis.
        # Instead, use the column space or row space directly (to find their respective bases).
        return self.calculate_column_space()

    def calculate_orthonormal_basis(self):
        """
        Calculate orthonormal basis using the output of calculate_basis() function. Use Gram-Schmidt process.
        """
        # Get the standard independent vectors (as rows, 1 vector per row)
        basis_vectors = self.calculate_basis()

        # Edge case: empty basis (e.g., zero matrix)
        if basis_vectors.size == 0:
            return basis_vectors
        
        orthonormal_basis = [] # Initialize list to store orthonormal basis vectors.
        TOL = 1e-10 # Tolerance for zero values
        
        for vector in basis_vectors:
            # Work on a copy to avoid side effects
            w = vector.astype(float).copy()
            
            # Formula: w = v - SUM( (v . u) * u )
            # where u = orthonormal_vector
            for u in orthonormal_basis:
                projection = np.dot(vector, u) * u
                w = w - projection
            
            # Normalize the vector (divide by length to get unit vector)
            norm = np.linalg.norm(w)
            if norm > TOL: # Avoid division by zero.
                u_new = w / norm
                orthonormal_basis.append(u_new)
                
        return np.array(orthonormal_basis)

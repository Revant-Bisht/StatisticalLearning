from math import pi
import numpy as np
from utils.base import BaseMatrixSolver
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class MyMatrixSolver(BaseMatrixSolver):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.gauss_jordan_matrix = self.gauss_jordan_elimination()
        self.determinant = self.calculate_determinant()
        
    def summary(self):
        report = f" Matrix has \
            RREF: {self.gauss_jordan_elimination()} \
            Dimensions: {self.matrix.shape} \
            Rank: {self.calculate_rank()} \
            Trace: {self.calculate_trace()} \
            Eigenvalues: {self.calculate_eigenvalues()} \
            Eigenvectors: {self.calculate_eigenvectors()} \
            Null Space: {self.calculate_null_space()} \
            Column Space: {self.calculate_column_space()} \
            Row Space: {self.calculate_row_space()} \
            Basis: {self.calculate_basis()} \
            Kernel: {self.calculate_kernel()} \
            Orthonormal Basis: {self.calculate_orthonormal_basis()} \
            Covariance Matrix: {self.calculate_covariance_matrix()} \
            Correlation Matrix: {self.calculate_correlation_matrix()} \
            Principal Components: {self.calculate_principal_components()} \
            Principal Component Analysis: {self.calculate_principal_component_analysis()} \
            Singular Value Decomposition: {self.calculate_singular_value_decomposition()} \
        "
        print(report)

    def gauss_jordan_elimination(self):
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
        A = self.matrix.astype(float).copy()
        rows, columns = A.shape
        TOL = 1e-10
        pivot_row = 0

        for col in range(columns):
            if pivot_row >= rows:
                break
            max_val_idx = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
            
            if abs(A[max_val_idx, col]) < TOL:
                continue

            if max_val_idx != pivot_row:
                A[[pivot_row, max_val_idx]] = A[[max_val_idx, pivot_row]]

            pivot = A[pivot_row, col]
            A[pivot_row] /= pivot

            for i in range(rows):
                if i != pivot_row:
                    factor = A[i, col]
                    if abs(factor) > 0:
                        A[i] = A[i] - factor * A[pivot_row]
            pivot_row += 1
        
        A[np.abs(A) < TOL] = 0.0
        return A

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

    def calculate_inverse(self):
        pass

    def calculate_rank(self):
        pass

    def calculate_trace(self):
        pass

    def calculate_eigenvalues(self):
        pass

    def calculate_eigenvectors(self):
        pass

    def calculate_null_space(self):
        pass

    def calculate_column_space(self):
        pass

    def calculate_row_space(self):
        pass

    def calculate_basis(self):
        pass

    def calculate_kernel(self):
        pass

    def calculate_orthonormal_basis(self):
        pass

    def calculate_covariance_matrix(self):
        pass

    def calculate_correlation_matrix(self):
        pass

    def calculate_principal_components(self):
        pass

    def calculate_principal_component_analysis(self):
        pass

    def calculate_singular_value_decomposition(self):
        pass
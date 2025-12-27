import numpy as np
from abc import ABC, abstractmethod

class BaseMatrixSolver(ABC):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @abstractmethod
    def gauss_jordan_elimination(self):
        pass

    @abstractmethod
    def calculate_determinant(self):
        pass

    @abstractmethod
    def calculate_inverse(self):
        pass

    @abstractmethod
    def calculate_rank(self):
        pass

    @abstractmethod
    def calculate_trace(self):
        pass

    @abstractmethod
    def calculate_eigenvalues(self):
        pass

    @abstractmethod
    def calculate_eigenvectors(self):
        pass

    @abstractmethod
    def calculate_null_space(self):
        pass

    @abstractmethod
    def calculate_column_space(self):
        pass

    @abstractmethod
    def calculate_row_space(self):
        pass

    @abstractmethod
    def calculate_basis(self):
        pass

    @abstractmethod
    def calculate_kernel(self):
        pass

    @abstractmethod
    def calculate_orthonormal_basis(self):
        pass

    @abstractmethod
    def calculate_covariance_matrix(self):
        pass

    @abstractmethod
    def calculate_correlation_matrix(self):
        pass

    @abstractmethod
    def calculate_principal_components(self):
        pass

    @abstractmethod
    def calculate_principal_component_analysis(self):
        pass

    @abstractmethod
    def calculate_singular_value_decomposition(self):
        pass
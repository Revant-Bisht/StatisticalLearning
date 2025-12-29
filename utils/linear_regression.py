from utils.matrices import MyMatrixSolver
import numpy as np  
import pandas as pd

def calculate_beta(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate Beta (coefficients) to match target matrix 'Y' using the data matrix 'X'.
    Formula: Beta = (X^T * X)^-1 * X^T * Y
    
    Args:
        X (np.ndarray): data matrix (N x p). N = samples, p = features.
        Y (np.ndarray): target matrix (N x m). N = samples, m = target variables.
        
    Returns:
        np.ndarray: coefficient vector Beta (p x m).
    """
    
    # Calculate X Transpose (X^T)
    X_T = X.T

    # Calculate Gram Matrix (X^T * X
    X_T_X = X_T @ X # (p x N) @ (N x p) = (p x p)

    # Calculate inverse of the Gram Matrix (X^T * X)^-1
    solver = MyMatrixSolver(X_T_X)
    X_T_X_inverse = solver.inverse

    # Check if inverse exists (in case of singular matrix)
    if X_T_X_inverse is None:
        raise ValueError("Matrix is singular and non-invertible. Features may be linearly dependent.")

    # Calculate X^T * Y (projection of Y onto the feature space)
    X_T_Y = X_T @ Y # (p x N) @ (N x m) = (p x m)

    # Final multiplication: (p x p)^-1 @ (p x m) = (p x m)
    # Resulting shape is (p x m).
    Beta = X_T_X_inverse @ X_T_Y # (p x p)^-1 @ (p x m) = (p x m)
    return Beta

def calculate_residual_sum_of_squares(X: np.ndarray, Y: np.ndarray, Beta: np.ndarray):
    """
    Calculate the residual sum of squares (RSS) to measure the difference between the predicted and actual values.
    Formula: RSS = (Y - X * Beta)^T * (Y - X * Beta)
    """
    # Calculate the predicted values (Y_hat)
    Y_hat = X @ Beta # (N x p) @ (p x m) = (N x m)
    # Calculate the residual values (Y - Y_hat)
    residuals = Y - Y_hat
    # Calculate the residual sum of squares (RSS)
    RSS_matrix = residuals.T @ residuals # (N x m)^T * (N x m) = (m x m)
    return RSS_matrix



# PCA.py
from utils.matrices import MyMatrixSolver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if the input data is valid
def check_input_data(X: np.ndarray):
    """
    Check if the input data is valid.
    Args:
        X (np.ndarray): Input data (n x p).
    Raises:
        ValueError: If the input data is not valid.
    """
    if not np.all(np.isfinite(X)): # if all elements are not finite, raise an error.
        raise ValueError(f"Input X contains NaN or Infinity.")
    if X.ndim != 2:
        raise ValueError(f"Input X must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError(f"Input X must have at least one sample.")
    if X.shape[1] == 0:
        raise ValueError(f"Input X must have at least one feature.")

def calculate_pca_parameters(
    X: np.ndarray, 
    k: int = None, 
    standardize: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Principal Components (Eigenvectors) and Mean from input data X.
    
    Steps:
    1. Calculate Mean vector (mu) (dimension: (p,)).
    2. Center data (X - mu) (dimension: n x p).
    3. Compute Covariance Matrix (cov_matrix) (dimension: p x p).
    4. Solve cov_matrix * v = lambda * v (Eigendecomposition) (dimension: p x p).
    5. Sort vectors by lambda (variance) (dimension: p x 1) (descending order).
    6. Keep only the top 'k' principal components (dimension: p x k) (default: all) (dimensionality reduction).
    
    Args:
        X (np.ndarray): Input data (n x p).
        k (int): Number of principal components to keep.
        standardize (bool): Whether to standardize the data before calculating the covariance matrix.
    Returns:
        tuple: (principal_components, explained_variance, mean_vector, std_vector)
               - principal_components (W): Matrix of sorted eigenvectors (p x k)
               - explained_variance (lambda): Array of sorted eigenvalues (k,)
               - mean_vector (mu): Array of column means (p,)
               - std_vector (sigma): Array of column standard deviations - None if not standardizing, (p,) otherwise.
    """

    check_input_data(X)
    n, p = X.shape

    # If k is None, keep all components.
    if k is None:
        k = p

    # Calculate Mean and center the data.
    mean_vector = np.mean(X, axis=0) # column means (dimension: (p,)). One mean for each feature.
    X_centered = X - mean_vector # NumPy handles this by broadcasting.

    # Optional standardization.
    if standardize:
        # ddof = 1 for sample standard deviation (n - 1 degrees of freedom).
        std_vector = np.std(X_centered, axis=0, ddof=1) # column standard deviations (dimension: (p,)). One std for each feature.
        X_processed = X_centered / std_vector # NumPy handles this by broadcasting.

    else:
        std_vector = None
        X_processed = X_centered

    # Use MyMatrixSolver to compute the covariance matrix from the data.
    data_solver = MyMatrixSolver(X_processed)

    # Covariance matrix (use solver for efficient computation).
    cov_matrix = data_solver.calculate_covariance_matrix() # dimension: (p x p).
    if cov_matrix is None:
        raise ValueError("Covariance matrix is None. Cannot compute PCA parameters.")

    # Eigen Decomposition of the covariance matrix (not the data matrix!)
    # Reason: PCA finds directions of maximum variance. The covariance matrix Σ captures
    # how features vary together. Its eigenvectors point in directions of maximum variance,
    # and its eigenvalues quantify how much variance exists along each direction.
    # Mathematically: Σv = λv, where v is a principal component and λ is the variance it explains.
    cov_solver = MyMatrixSolver(cov_matrix)

    eigenvectors = cov_solver.eigenvectors # dimension: (p x p).
    if eigenvectors is None or eigenvectors.shape[0] != p:
        raise ValueError("Eigenvectors are None or have the wrong shape. Cannot compute PCA parameters.")

    eigenvalues = cov_solver.eigenvalues # dimension: (p,).
    # Check if eigenvalues are negative, if so, raise an error. Cannot have negative variance.
    if np.any(eigenvalues < 0):
        raise ValueError("Eigenvalues are negative. Cannot compute PCA parameters.")

    # Make eigenvetors columns for easier manipulation.
    eigenvectors = eigenvectors.T # dimension is still (p x p), but now each column is an eigenvector.

    # Sort eigenvalues in descending order (top eigenvalues capture/explain the highest magnitude of variance)
    sorted_indices = np.argsort(eigenvalues)[::-1] # [::-1] reverses the array.

    sorted_eigenvalues = eigenvalues[sorted_indices] # returns sorted eigenvalues in descending order.
    sorted_eigenvectors = eigenvectors[:, sorted_indices] # returns sorted eigenvectors in descending order.

    # Select only the top 'k' components (dimensionality reduction).
    principal_components = sorted_eigenvectors[:, :k]  # Select first k columns, giving (p x k)
    explained_variance = sorted_eigenvalues[:k] # Select first k eigenvalues, giving (k,).

    return principal_components, explained_variance, mean_vector, std_vector


def apply_pca_transform(
    X: np.ndarray,
    principal_components: np.ndarray,
    mean_vector: np.ndarray,
    std_vector: np.ndarray = None
    ) -> np.ndarray:
    """
    Projects data X onto the Principal Components.
    
    Formula: X_new = (X - mu) @ W
    
    Args:
        X (np.ndarray): New data to transform (N x p).
        principal_components (np.ndarray): The sorted eigenvectors W calculated in training phase (p x k).
        mean_vector (np.ndarray): The mean mu calculated in training phase (p,).
        std_vector (np.ndarray): The std calculated in training phase (p,). Required if standardizing was used in training phase.
        
    Returns:
        np.ndarray: Transformed data (N x k).
    """
    check_input_data(X)
    
    # Check dimensions
    if X.shape[1] != principal_components.shape[0]:
        raise ValueError(f"Dimension Mismatch: Data has {X.shape[1]} features, but PCA expects {principal_components.shape[0]}.")

    # Center the data using the mean_vector calculated in training phase.
    X_centered = X - mean_vector

    if std_vector is not None:
        X_centered = X_centered / std_vector

    # Project the data onto the principal components.
    X_transformed = X_centered @ principal_components # (N x p) @ (p x k) -> (N x k)

    return X_transformed


def explained_variance_summary(explained_variance: np.ndarray) -> pd.DataFrame:
    """
    Creates a summary DataFrame of the explained variance for each principal component.
    
    Args:
        explained_variance (np.ndarray): The explained variance/eigenvalues (k,).
        
    Returns:
        pd.DataFrame: Summary with columns:
            - 'Principal Component': Component number (1, 2, 3, ...)
            - 'Eigenvalue': Raw eigenvalue (explained variance)
            - 'Variance Ratio': Proportion of total variance (0.01 to 1.00)
            - 'Variance %': Percentage of total variance (0.01% to 100.00%)
            - 'Cumulative %': Cumulative percentage of variance explained (0.01% to 100.00%)
    """
    total_variance = np.sum(explained_variance)
    variance_ratio = explained_variance / total_variance
    variance_percentage = variance_ratio * 100
    cumulative_percentage = np.cumsum(variance_percentage)
    
    # Pass the data to a DataFrame as a dictionary. 
    # Key = column name, Value = data.
    summary_df = pd.DataFrame({
        'Principal Component': np.arange(1, len(explained_variance) + 1),
        'Eigenvalue': explained_variance,
        'Variance Ratio': variance_ratio,
        'Variance %': variance_percentage,
        'Cumulative %': cumulative_percentage
    })
    
    return summary_df


def scree_plot(explained_variance: np.ndarray) -> None:
    """
    Plots the scree plot of the explained variance (eigenvalues/lambda) vs the principal component.
    Args:
        explained_variance (np.ndarray): The explained variance (k,).
    Returns:
        None
    """
    x = np.arange(1, len(explained_variance) + 1) # x-axis is the principal component number (1, 2, 3, ...).
    variance_ratio = explained_variance / np.sum(explained_variance) * 100 # % of variance explained by each PC.
    plt.xticks(x) # integer ticks for the x-axis.
    plt.plot(x, variance_ratio, 'bo-') # 'bo-' means blue dots connected by a line.
    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance Ratio (%)')
    plt.title('Scree Plot of Explained Variance')
    plt.grid(True, alpha = 0.4) # add grid to the plot. alpha = transparency (0.0 to 1.0).
    plt.show()
    return None
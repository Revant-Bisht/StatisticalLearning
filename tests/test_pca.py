import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from utils.PCA import (
    calculate_pca_parameters,
    apply_pca_transform,
    explained_variance_summary,
    check_input_data
)


class TestPCAvsSklearn:
    """Tests comparing custom PCA implementation with sklearn's PCA."""

    @pytest.fixture
    def sample_data_small(self):
        """Small 2D dataset for basic testing."""
        np.random.seed(42)
        return np.random.randn(50, 4)

    @pytest.fixture
    def sample_data_large(self):
        """Larger dataset for more comprehensive testing."""
        np.random.seed(123)
        return np.random.randn(200, 10)

    @pytest.fixture
    def correlated_data(self):
        """Dataset with correlated features (ideal for PCA)."""
        np.random.seed(99)
        n_samples = 100
        # Create correlated features
        x1 = np.random.randn(n_samples)
        x2 = x1 * 0.8 + np.random.randn(n_samples) * 0.2  # Highly correlated with x1
        x3 = np.random.randn(n_samples)  # Independent
        x4 = x3 * 0.5 + np.random.randn(n_samples) * 0.5  # Moderately correlated with x3
        return np.column_stack([x1, x2, x3, x4])

    def test_explained_variance_matches_sklearn(self, sample_data_small):
        """Explained variance (eigenvalues) should match sklearn."""
        X = sample_data_small
        
        # Custom PCA
        principal_components, explained_variance, mean_vector, _ = calculate_pca_parameters(X)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA()
        sklearn_pca.fit(X)
        
        # Compare explained variances (eigenvalues)
        # Note: sklearn uses n-1 denominator for variance, we need to check consistency
        np.testing.assert_array_almost_equal(
            explained_variance, 
            sklearn_pca.explained_variance_, 
            decimal=5,
            err_msg="Explained variance does not match sklearn"
        )

    def test_explained_variance_ratio_matches_sklearn(self, sample_data_small):
        """Explained variance ratio should match sklearn."""
        X = sample_data_small
        
        # Custom PCA
        _, explained_variance, _, _ = calculate_pca_parameters(X)
        custom_ratio = explained_variance / np.sum(explained_variance)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA()
        sklearn_pca.fit(X)
        
        np.testing.assert_array_almost_equal(
            custom_ratio, 
            sklearn_pca.explained_variance_ratio_, 
            decimal=5,
            err_msg="Explained variance ratio does not match sklearn"
        )

    def test_principal_components_match_sklearn(self, sample_data_small):
        """Principal components (eigenvectors) should match sklearn up to sign."""
        X = sample_data_small
        
        # Custom PCA
        principal_components, _, _, _ = calculate_pca_parameters(X)
        
        # Sklearn PCA - components are stored as rows, so transpose to compare
        sklearn_pca = SklearnPCA()
        sklearn_pca.fit(X)
        sklearn_components = sklearn_pca.components_.T  # Convert to column vectors
        
        # Eigenvectors can differ by sign, so compare absolute values
        np.testing.assert_array_almost_equal(
            np.abs(principal_components), 
            np.abs(sklearn_components), 
            decimal=5,
            err_msg="Principal components do not match sklearn (up to sign)"
        )

    def test_mean_vector_matches_sklearn(self, sample_data_small):
        """Mean vector should match sklearn's mean_."""
        X = sample_data_small
        
        # Custom PCA
        _, _, mean_vector, _ = calculate_pca_parameters(X)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA()
        sklearn_pca.fit(X)
        
        np.testing.assert_array_almost_equal(
            mean_vector, 
            sklearn_pca.mean_, 
            decimal=10,
            err_msg="Mean vector does not match sklearn"
        )

    def test_transform_matches_sklearn_up_to_sign(self, sample_data_small):
        """Transformed data should match sklearn (up to sign flips per component)."""
        X = sample_data_small
        k = 3  # Keep 3 components
        
        # Custom PCA
        principal_components, _, mean_vector, _ = calculate_pca_parameters(X, k=k)
        X_custom = apply_pca_transform(X, principal_components, mean_vector)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA(n_components=k)
        X_sklearn = sklearn_pca.fit_transform(X)
        
        # Compare absolute values (eigenvectors may have opposite signs)
        np.testing.assert_array_almost_equal(
            np.abs(X_custom), 
            np.abs(X_sklearn), 
            decimal=5,
            err_msg="Transformed data does not match sklearn (up to sign)"
        )

    def test_dimensionality_reduction(self, sample_data_large):
        """Test that dimensionality reduction works correctly."""
        X = sample_data_large
        k = 5  # Reduce from 10 to 5 dimensions
        
        # Custom PCA
        principal_components, explained_variance, mean_vector, _ = calculate_pca_parameters(X, k=k)
        X_transformed = apply_pca_transform(X, principal_components, mean_vector)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA(n_components=k)
        X_sklearn = sklearn_pca.fit_transform(X)
        
        # Check output dimensions
        assert X_transformed.shape == (200, 5), f"Expected (200, 5), got {X_transformed.shape}"
        assert X_transformed.shape == X_sklearn.shape
        
        # Compare results
        np.testing.assert_array_almost_equal(
            np.abs(X_transformed), 
            np.abs(X_sklearn), 
            decimal=5
        )

    def test_correlated_data_variance_capture(self, correlated_data):
        """PCA should capture most variance in first few components for correlated data."""
        X = correlated_data
        
        # Custom PCA
        _, explained_variance, _, _ = calculate_pca_parameters(X)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA()
        sklearn_pca.fit(X)
        
        # Both should agree that first 2 components capture most variance
        custom_cumsum = np.cumsum(explained_variance) / np.sum(explained_variance)
        sklearn_cumsum = np.cumsum(sklearn_pca.explained_variance_ratio_)
        
        np.testing.assert_array_almost_equal(
            custom_cumsum, 
            sklearn_cumsum, 
            decimal=5
        )
        
        # For correlated data, first 2 components should capture > 70% variance
        assert custom_cumsum[1] > 0.7, "First 2 components should capture significant variance"

    def test_single_component_pca(self, sample_data_small):
        """Test PCA with k=1 component."""
        X = sample_data_small
        
        # Custom PCA
        principal_components, explained_variance, mean_vector, _ = calculate_pca_parameters(X, k=1)
        X_custom = apply_pca_transform(X, principal_components, mean_vector)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA(n_components=1)
        X_sklearn = sklearn_pca.fit_transform(X)
        
        assert X_custom.shape == (50, 1)
        np.testing.assert_array_almost_equal(
            np.abs(X_custom), 
            np.abs(X_sklearn), 
            decimal=5
        )

    def test_full_components_equals_original_variance(self, sample_data_small):
        """Sum of all explained variances should equal total data variance."""
        X = sample_data_small
        
        # Custom PCA
        _, explained_variance, _, _ = calculate_pca_parameters(X)
        
        # Total variance from data (sum of variances of each feature)
        total_data_variance = np.var(X, axis=0, ddof=1).sum()
        total_pca_variance = np.sum(explained_variance)
        
        np.testing.assert_almost_equal(
            total_pca_variance, 
            total_data_variance, 
            decimal=5,
            err_msg="Total explained variance should equal total data variance"
        )


class TestPCAWithStandardization:
    """Tests for PCA with standardization (correlation-based PCA)."""

    @pytest.fixture
    def varied_scale_data(self):
        """Dataset with features on different scales."""
        np.random.seed(42)
        n_samples = 100
        # Features with very different scales
        x1 = np.random.randn(n_samples) * 1000  # Large scale
        x2 = np.random.randn(n_samples) * 0.01   # Small scale
        x3 = np.random.randn(n_samples) * 50     # Medium scale
        return np.column_stack([x1, x2, x3])

    def test_standardized_pca_matches_sklearn_with_prescaling(self, varied_scale_data):
        """Standardized PCA should match sklearn PCA on pre-standardized data."""
        X = varied_scale_data
        
        # Custom PCA with standardization
        principal_components, explained_variance, mean_vector, std_vector = calculate_pca_parameters(
            X, standardize=True
        )
        
        # Sklearn: manually standardize then apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        sklearn_pca = SklearnPCA()
        sklearn_pca.fit(X_scaled)
        
        # Compare explained variance ratios
        custom_ratio = explained_variance / np.sum(explained_variance)
        np.testing.assert_array_almost_equal(
            custom_ratio, 
            sklearn_pca.explained_variance_ratio_, 
            decimal=4,
            err_msg="Standardized PCA variance ratio should match sklearn on scaled data"
        )

    def test_standardization_produces_std_vector(self, varied_scale_data):
        """When standardizing, std_vector should be returned."""
        X = varied_scale_data
        
        _, _, _, std_vector = calculate_pca_parameters(X, standardize=True)
        
        assert std_vector is not None, "std_vector should not be None when standardizing"
        assert len(std_vector) == X.shape[1], "std_vector should have length equal to number of features"

    def test_non_standardization_returns_none_std(self, varied_scale_data):
        """When not standardizing, std_vector should be None."""
        X = varied_scale_data
        
        _, _, _, std_vector = calculate_pca_parameters(X, standardize=False)
        
        assert std_vector is None, "std_vector should be None when not standardizing"


class TestPCAInputValidation:
    """Tests for input validation in PCA functions."""

    def test_nan_input_raises_error(self):
        """Input with NaN should raise ValueError."""
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        with pytest.raises(ValueError, match="NaN or Infinity"):
            check_input_data(X)

    def test_inf_input_raises_error(self):
        """Input with Infinity should raise ValueError."""
        X = np.array([[1, 2], [np.inf, 4], [5, 6]])
        with pytest.raises(ValueError, match="NaN or Infinity"):
            check_input_data(X)

    def test_1d_input_raises_error(self):
        """1D array should raise ValueError."""
        X = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="2D array"):
            check_input_data(X)

    def test_empty_samples_raises_error(self):
        """Input with 0 samples should raise ValueError."""
        X = np.empty((0, 3))
        with pytest.raises(ValueError, match="at least one sample"):
            check_input_data(X)

    def test_empty_features_raises_error(self):
        """Input with 0 features should raise ValueError."""
        X = np.empty((5, 0))
        with pytest.raises(ValueError, match="at least one feature"):
            check_input_data(X)

    def test_dimension_mismatch_in_transform(self):
        """Transform should fail if data dimensions don't match PCA components."""
        X_train = np.random.randn(50, 4)
        X_test = np.random.randn(20, 5)  # Different number of features
        
        principal_components, _, mean_vector, _ = calculate_pca_parameters(X_train)
        
        with pytest.raises(ValueError, match="Dimension Mismatch"):
            apply_pca_transform(X_test, principal_components, mean_vector)


class TestExplainedVarianceSummary:
    """Tests for the explained variance summary function."""

    def test_summary_columns(self):
        """Summary DataFrame should have correct columns."""
        explained_variance = np.array([3.0, 2.0, 1.0])
        summary = explained_variance_summary(explained_variance)
        
        expected_cols = ['Principal Component', 'Eigenvalue', 'Variance Ratio', 
                         'Variance %', 'Cumulative %']
        assert list(summary.columns) == expected_cols

    def test_summary_cumulative_percentage(self):
        """Cumulative percentage should sum to 100%."""
        explained_variance = np.array([3.0, 2.0, 1.0])
        summary = explained_variance_summary(explained_variance)
        
        assert summary['Cumulative %'].iloc[-1] == pytest.approx(100.0)

    def test_summary_variance_ratios(self):
        """Variance ratios should be correctly calculated."""
        explained_variance = np.array([6.0, 3.0, 1.0])  # Total = 10
        summary = explained_variance_summary(explained_variance)
        
        expected_ratios = np.array([0.6, 0.3, 0.1])
        np.testing.assert_array_almost_equal(
            summary['Variance Ratio'].values, 
            expected_ratios
        )

    def test_summary_principal_component_numbers(self):
        """Principal component numbers should be 1-indexed."""
        explained_variance = np.array([3.0, 2.0, 1.0, 0.5])
        summary = explained_variance_summary(explained_variance)
        
        expected_pc = [1, 2, 3, 4]
        assert list(summary['Principal Component']) == expected_pc


class TestPCATransformOnNewData:
    """Tests for applying PCA transform to new/test data."""

    @pytest.fixture
    def train_test_split(self):
        """Create train and test datasets."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(30, 5)
        return X_train, X_test

    def test_transform_new_data_matches_sklearn(self, train_test_split):
        """Transforming new data should match sklearn's transform."""
        X_train, X_test = train_test_split
        k = 3
        
        # Custom PCA
        principal_components, _, mean_vector, _ = calculate_pca_parameters(X_train, k=k)
        X_test_custom = apply_pca_transform(X_test, principal_components, mean_vector)
        
        # Sklearn PCA
        sklearn_pca = SklearnPCA(n_components=k)
        sklearn_pca.fit(X_train)
        X_test_sklearn = sklearn_pca.transform(X_test)
        
        np.testing.assert_array_almost_equal(
            np.abs(X_test_custom), 
            np.abs(X_test_sklearn), 
            decimal=5,
            err_msg="Transform on new data should match sklearn"
        )

    def test_transform_preserves_sample_count(self, train_test_split):
        """Number of samples should be preserved after transform."""
        X_train, X_test = train_test_split
        k = 2
        
        principal_components, _, mean_vector, _ = calculate_pca_parameters(X_train, k=k)
        X_test_transformed = apply_pca_transform(X_test, principal_components, mean_vector)
        
        assert X_test_transformed.shape[0] == X_test.shape[0]
        assert X_test_transformed.shape[1] == k


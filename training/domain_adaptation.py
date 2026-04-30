"""Domain Adaptation techniques for cross-domain audio deepfake detection.

Implements:
1. CORAL (CORrelation ALignment) - Aligns covariance matrices
2. MMD (Maximum Mean Discrepancy) - Kernel-based distribution alignment
3. Combined approaches for feature transformation

Reference:
- CORAL: Sun & Saenko, "Return of Frustratingly Easy Domain Adaptation" (2016)
- MMD: Gretton et al., "A Kernel Two-Sample Test" (2012)
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings


class CORALTransformer(BaseEstimator, TransformerMixin):
    """CORAL: CORrelation ALignment for domain adaptation.

    Aligns the covariance of source features to match target domain covariance.
    This is a simple, unsupervised domain adaptation technique.

    The transformation A is computed such that:
        A^T * Cov_source * A ≈ Cov_target

    At inference, source features are transformed: X_aligned = X_source @ A
    """

    def __init__(self, reg: float = 1e-6):
        """Initialize CORAL transformer.

        Args:
            reg: Regularization for covariance matrix inversion
        """
        self.reg = reg
        self.A_ = None  # Transformation matrix
        self.source_mean_ = None
        self.target_mean_ = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'CORALTransformer':
        """Compute CORAL transformation from source to target domain.

        Args:
            X_source: Source domain features (n_source, n_features)
            X_target: Target domain features (n_target, n_features)

        Returns:
            self
        """
        # Center the data
        self.source_mean_ = X_source.mean(axis=0)
        self.target_mean_ = X_target.mean(axis=0)

        X_s = X_source - self.source_mean_
        X_t = X_target - self.target_mean_

        # Compute covariance matrices
        n_s = X_s.shape[0]
        n_t = X_t.shape[0]

        C_s = (X_s.T @ X_s) / (n_s - 1) + self.reg * np.eye(X_s.shape[1])
        C_t = (X_t.T @ X_t) / (n_t - 1) + self.reg * np.eye(X_t.shape[1])

        # Compute whitening transformation for source
        # C_s = U_s @ D_s @ U_s^T
        D_s, U_s = linalg.eigh(C_s)
        D_s = np.maximum(D_s, self.reg)  # Ensure positive

        # Compute coloring transformation for target
        D_t, U_t = linalg.eigh(C_t)
        D_t = np.maximum(D_t, self.reg)

        # CORAL transformation: whiten source, then color with target
        # A = C_s^(-1/2) @ C_t^(1/2)
        whitening = U_s @ np.diag(1.0 / np.sqrt(D_s)) @ U_s.T
        coloring = U_t @ np.diag(np.sqrt(D_t)) @ U_t.T

        self.A_ = whitening @ coloring

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using CORAL alignment.

        Args:
            X: Features to transform (n_samples, n_features)

        Returns:
            Transformed features
        """
        if self.A_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        # Center using source mean, transform, then shift to target mean
        X_centered = X - self.source_mean_
        X_transformed = X_centered @ self.A_
        X_aligned = X_transformed + self.target_mean_

        return X_aligned

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """Fit and transform source features.

        Args:
            X_source: Source domain features
            X_target: Target domain features

        Returns:
            Transformed source features
        """
        self.fit(X_source, X_target)
        return self.transform(X_source)


class MMDTransformer(BaseEstimator, TransformerMixin):
    """MMD-based domain adaptation via optimal transport.

    Uses iterative gradient descent to find a transformation that
    minimizes MMD between transformed source and target distributions.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: Optional[float] = None,
        learning_rate: float = 0.01,
        n_iterations: int = 100,
        reg: float = 1e-4,
    ):
        """Initialize MMD transformer.

        Args:
            kernel: Kernel type ('rbf', 'linear')
            gamma: RBF kernel bandwidth (None = median heuristic)
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of optimization iterations
            reg: Regularization strength
        """
        self.kernel = kernel
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg = reg
        self.W_ = None  # Transformation matrix
        self.b_ = None  # Bias term

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'rbf':
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
            dists = X_sq + Y_sq.T - 2 * X @ Y.T
            return np.exp(-gamma * dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_mmd(self, X_s: np.ndarray, X_t: np.ndarray, gamma: float) -> float:
        """Compute MMD^2 between two distributions."""
        K_ss = self._compute_kernel(X_s, X_s, gamma)
        K_tt = self._compute_kernel(X_t, X_t, gamma)
        K_st = self._compute_kernel(X_s, X_t, gamma)

        n_s, n_t = X_s.shape[0], X_t.shape[0]

        mmd = (K_ss.sum() / (n_s * n_s) +
               K_tt.sum() / (n_t * n_t) -
               2 * K_st.sum() / (n_s * n_t))

        return mmd

    def _median_heuristic(self, X: np.ndarray) -> float:
        """Compute gamma using median heuristic."""
        # Sample for efficiency
        n = min(1000, X.shape[0])
        idx = np.random.choice(X.shape[0], n, replace=False)
        X_sample = X[idx]

        dists = np.sum((X_sample[:, None] - X_sample[None, :]) ** 2, axis=2)
        median_dist = np.median(dists[np.triu_indices(n, k=1)])

        return 1.0 / (median_dist + 1e-8)

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'MMDTransformer':
        """Learn transformation to minimize MMD.

        Args:
            X_source: Source domain features
            X_target: Target domain features

        Returns:
            self
        """
        n_features = X_source.shape[1]

        # Initialize transformation as identity + small noise
        self.W_ = np.eye(n_features) + 0.01 * np.random.randn(n_features, n_features)
        self.b_ = np.zeros(n_features)

        # Compute gamma if not provided
        gamma = self.gamma
        if gamma is None:
            gamma = self._median_heuristic(np.vstack([X_source, X_target]))
        self.gamma_ = gamma

        # Standardize
        self.source_scaler_ = StandardScaler()
        self.target_scaler_ = StandardScaler()

        X_s = self.source_scaler_.fit_transform(X_source)
        X_t = self.target_scaler_.fit_transform(X_target)

        # Gradient descent to minimize MMD
        best_mmd = float('inf')
        best_W = self.W_.copy()
        best_b = self.b_.copy()

        for i in range(self.n_iterations):
            # Transform source
            X_s_trans = X_s @ self.W_ + self.b_

            # Compute MMD
            mmd = self._compute_mmd(X_s_trans, X_t, gamma)

            if mmd < best_mmd:
                best_mmd = mmd
                best_W = self.W_.copy()
                best_b = self.b_.copy()

            # Compute gradients numerically (simple approach)
            eps = 1e-5
            grad_W = np.zeros_like(self.W_)
            grad_b = np.zeros_like(self.b_)

            for j in range(n_features):
                for k in range(n_features):
                    self.W_[j, k] += eps
                    X_s_pert = X_s @ self.W_ + self.b_
                    mmd_pert = self._compute_mmd(X_s_pert, X_t, gamma)
                    grad_W[j, k] = (mmd_pert - mmd) / eps
                    self.W_[j, k] -= eps

                self.b_[j] += eps
                X_s_pert = X_s @ self.W_ + self.b_
                mmd_pert = self._compute_mmd(X_s_pert, X_t, gamma)
                grad_b[j] = (mmd_pert - mmd) / eps
                self.b_[j] -= eps

            # Update with regularization toward identity
            self.W_ -= self.learning_rate * (grad_W + self.reg * (self.W_ - np.eye(n_features)))
            self.b_ -= self.learning_rate * (grad_b + self.reg * self.b_)

        # Use best found
        self.W_ = best_W
        self.b_ = best_b

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using learned MMD alignment."""
        if self.W_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_scaled = self.source_scaler_.transform(X)
        X_transformed = X_scaled @ self.W_ + self.b_
        X_aligned = self.target_scaler_.inverse_transform(X_transformed)

        return X_aligned


class SubspaceCORAL(BaseEstimator, TransformerMixin):
    """Subspace-based CORAL that first projects to a shared subspace.

    This can be more robust when source and target have different
    intrinsic dimensionalities.
    """

    def __init__(self, n_components: int = 10, reg: float = 1e-6):
        """Initialize Subspace CORAL.

        Args:
            n_components: Number of PCA components for subspace
            reg: Regularization parameter
        """
        self.n_components = n_components
        self.reg = reg

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'SubspaceCORAL':
        """Fit subspace CORAL transformation."""
        from sklearn.decomposition import PCA

        # Fit PCA on combined data
        X_combined = np.vstack([X_source, X_target])
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_combined)

        # Project to subspace
        X_s_pca = self.pca_.transform(X_source)
        X_t_pca = self.pca_.transform(X_target)

        # Apply CORAL in subspace
        self.coral_ = CORALTransformer(reg=self.reg)
        self.coral_.fit(X_s_pca, X_t_pca)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features via subspace CORAL."""
        X_pca = self.pca_.transform(X)
        X_aligned = self.coral_.transform(X_pca)
        return X_aligned


class DomainAdaptiveClassifier:
    """Wrapper that applies domain adaptation before classification.

    Supports:
    - CORAL adaptation
    - MMD adaptation
    - No adaptation (baseline)
    - Target-normalized (simple mean/std alignment)
    """

    def __init__(
        self,
        classifier,
        adaptation_method: str = 'coral',
        **adaptation_kwargs
    ):
        """Initialize domain-adaptive classifier.

        Args:
            classifier: Base classifier (must have fit/predict/predict_proba)
            adaptation_method: 'coral', 'mmd', 'target_norm', or 'none'
            **adaptation_kwargs: Arguments for adaptation transformer
        """
        self.classifier = classifier
        self.adaptation_method = adaptation_method
        self.adaptation_kwargs = adaptation_kwargs
        self.adapter_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
    ) -> 'DomainAdaptiveClassifier':
        """Fit adapter and classifier.

        Args:
            X_source: Source domain features (labeled)
            y_source: Source domain labels
            X_target: Target domain features (unlabeled, for adaptation)

        Returns:
            self
        """
        # Create and fit adapter
        if self.adaptation_method == 'coral':
            self.adapter_ = CORALTransformer(**self.adaptation_kwargs)
        elif self.adaptation_method == 'mmd':
            self.adapter_ = MMDTransformer(**self.adaptation_kwargs)
        elif self.adaptation_method == 'subspace_coral':
            self.adapter_ = SubspaceCORAL(**self.adaptation_kwargs)
        elif self.adaptation_method == 'target_norm':
            self.adapter_ = TargetNormTransformer()
        elif self.adaptation_method == 'none':
            self.adapter_ = None
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")

        # Fit adapter
        if self.adapter_ is not None:
            self.adapter_.fit(X_source, X_target)
            X_source_adapted = self.adapter_.transform(X_source)
        else:
            X_source_adapted = X_source

        # Fit classifier on adapted source data
        self.classifier.fit(X_source_adapted, y_source)
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted.")

        if self.adapter_ is not None:
            X_adapted = self.adapter_.transform(X)
        else:
            X_adapted = X

        return self.classifier.predict(X_adapted)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities on new data."""
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted.")

        if self.adapter_ is not None:
            X_adapted = self.adapter_.transform(X)
        else:
            X_adapted = X

        return self.classifier.predict_proba(X_adapted)


class TargetNormTransformer(BaseEstimator, TransformerMixin):
    """Simple target normalization: align mean and std to target domain."""

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'TargetNormTransformer':
        """Compute normalization parameters."""
        self.source_mean_ = X_source.mean(axis=0)
        self.source_std_ = X_source.std(axis=0) + 1e-8
        self.target_mean_ = X_target.mean(axis=0)
        self.target_std_ = X_target.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to target normalization."""
        X_normalized = (X - self.source_mean_) / self.source_std_
        X_target_scaled = X_normalized * self.target_std_ + self.target_mean_
        return X_target_scaled


def compute_domain_discrepancy(
    X_source: np.ndarray,
    X_target: np.ndarray,
    method: str = 'mmd'
) -> float:
    """Compute domain discrepancy metric.

    Args:
        X_source: Source features
        X_target: Target features
        method: 'mmd' or 'coral_dist'

    Returns:
        Discrepancy score (lower = more similar)
    """
    if method == 'mmd':
        transformer = MMDTransformer()
        gamma = transformer._median_heuristic(np.vstack([X_source, X_target]))
        return transformer._compute_mmd(X_source, X_target, gamma)
    elif method == 'coral_dist':
        # Frobenius norm of covariance difference
        C_s = np.cov(X_source.T)
        C_t = np.cov(X_target.T)
        return np.linalg.norm(C_s - C_t, 'fro')
    else:
        raise ValueError(f"Unknown method: {method}")

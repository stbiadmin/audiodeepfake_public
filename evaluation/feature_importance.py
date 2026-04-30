"""Feature importance analysis for audio deepfake detection.

Combines multiple importance methods: XGBoost built-in, SHAP, and permutation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance


def get_xgboost_importance(
    model: XGBClassifier,
    feature_names: List[str],
    importance_type: str = 'gain',
) -> Dict[str, float]:
    """Get XGBoost built-in feature importance.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        importance_type: Type of importance ('gain', 'weight', 'cover')

    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Use sklearn interface for simplicity
    importances = model.feature_importances_
    result = dict(zip(feature_names, importances.tolist()))

    # Normalize to sum to 1
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def get_shap_importance(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 1000,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Get SHAP-based feature importance.

    Args:
        model: Trained XGBoost model
        X: Feature DataFrame for SHAP calculation
        feature_names: List of feature names
        max_samples: Maximum samples to use for SHAP

    Returns:
        Tuple of (importance_dict, shap_values_array)
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping SHAP importance.")
        return {}, np.array([])

    if feature_names is None:
        feature_names = list(X.columns)

    # Subsample if needed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    result = dict(zip(feature_names, mean_shap.tolist()))

    # Normalize
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)), shap_values


def get_permutation_importance(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Get permutation-based feature importance.

    Args:
        model: Trained model
        X: Feature DataFrame
        y: True labels
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if feature_names is None:
        feature_names = list(X.columns)

    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='f1',
    )

    # Use mean importance
    importances = perm_importance.importances_mean

    # Clip negative values to 0 (can happen with noise)
    importances = np.clip(importances, 0, None)

    result = dict(zip(feature_names, importances.tolist()))

    # Normalize
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def get_combined_importance(
    xgboost_imp: Dict[str, float],
    shap_imp: Dict[str, float],
    perm_imp: Dict[str, float],
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> Dict[str, float]:
    """Combine multiple importance measures with weighted average.

    Args:
        xgboost_imp: XGBoost importance
        shap_imp: SHAP importance
        perm_imp: Permutation importance
        weights: Weights for (xgboost, shap, permutation)

    Returns:
        Combined importance dictionary
    """
    all_features = set(xgboost_imp.keys()) | set(shap_imp.keys()) | set(perm_imp.keys())

    combined = {}
    for feature in all_features:
        xgb = xgboost_imp.get(feature, 0)
        shp = shap_imp.get(feature, 0)
        prm = perm_imp.get(feature, 0)

        combined[feature] = weights[0] * xgb + weights[1] * shp + weights[2] * prm

    # Normalize
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}

    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods."""

    def __init__(
        self,
        model: XGBClassifier,
        feature_names: List[str],
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    ):
        """Initialize analyzer.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            weights: Weights for combining methods (xgboost, shap, permutation)
        """
        self.model = model
        self.feature_names = feature_names
        self.weights = weights

        self.xgboost_importance_: Optional[Dict[str, float]] = None
        self.shap_importance_: Optional[Dict[str, float]] = None
        self.permutation_importance_: Optional[Dict[str, float]] = None
        self.combined_importance_: Optional[Dict[str, float]] = None
        self.shap_values_: Optional[np.ndarray] = None

    def analyze(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        compute_shap: bool = True,
        compute_permutation: bool = True,
    ) -> Dict[str, float]:
        """Run full importance analysis.

        Args:
            X: Feature DataFrame
            y: True labels
            compute_shap: Whether to compute SHAP values
            compute_permutation: Whether to compute permutation importance

        Returns:
            Combined importance dictionary
        """
        # XGBoost importance (always computed)
        self.xgboost_importance_ = get_xgboost_importance(
            self.model, self.feature_names
        )

        # SHAP importance
        if compute_shap:
            self.shap_importance_, self.shap_values_ = get_shap_importance(
                self.model, X, self.feature_names
            )
        else:
            self.shap_importance_ = {}
            self.shap_values_ = np.array([])

        # Permutation importance
        if compute_permutation:
            self.permutation_importance_ = get_permutation_importance(
                self.model, X, y, self.feature_names
            )
        else:
            self.permutation_importance_ = {}

        # Combined importance
        if self.shap_importance_ and self.permutation_importance_:
            self.combined_importance_ = get_combined_importance(
                self.xgboost_importance_,
                self.shap_importance_,
                self.permutation_importance_,
                self.weights,
            )
        else:
            self.combined_importance_ = self.xgboost_importance_

        return self.combined_importance_

    def get_importance_table(self) -> pd.DataFrame:
        """Get importance scores as a DataFrame.

        Returns:
            DataFrame with all importance methods as columns
        """
        data = {'feature': self.feature_names}

        if self.xgboost_importance_:
            data['xgboost'] = [self.xgboost_importance_.get(f, 0) for f in self.feature_names]

        if self.shap_importance_:
            data['shap'] = [self.shap_importance_.get(f, 0) for f in self.feature_names]

        if self.permutation_importance_:
            data['permutation'] = [self.permutation_importance_.get(f, 0) for f in self.feature_names]

        if self.combined_importance_:
            data['combined'] = [self.combined_importance_.get(f, 0) for f in self.feature_names]

        df = pd.DataFrame(data)
        return df.sort_values('combined' if 'combined' in df.columns else 'xgboost', ascending=False)

    def plot_importance(
        self,
        top_n: int = 15,
        save_path: Optional[Path] = None,
        title: str = "Feature Importance",
    ) -> None:
        """Plot feature importance bar chart.

        Args:
            top_n: Number of top features to show
            save_path: Path to save figure
            title: Plot title
        """
        if self.combined_importance_ is None:
            raise ValueError("Call analyze() first")

        # Get top features
        features = list(self.combined_importance_.keys())[:top_n]
        values = [self.combined_importance_[f] for f in features]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))

        ax.barh(y_pos, values, color='#2166ac', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        save_path: Optional[Path] = None,
        max_display: int = 15,
    ) -> None:
        """Plot SHAP summary plot.

        Args:
            X: Feature DataFrame
            save_path: Path to save figure
            max_display: Maximum features to display
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Cannot create summary plot.")
            return

        if self.shap_values_ is None or len(self.shap_values_) == 0:
            print("SHAP values not computed. Call analyze() with compute_shap=True")
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values_,
            X.iloc[:len(self.shap_values_)],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis results to dictionary."""
        return {
            'xgboost': self.xgboost_importance_,
            'shap': self.shap_importance_,
            'permutation': self.permutation_importance_,
            'combined': self.combined_importance_,
        }

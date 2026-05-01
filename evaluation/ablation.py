"""Ablation studies for feature analysis.

Experiments to understand feature contributions:
- Leave-one-out: Remove each feature and measure impact
- Incremental addition: Add features one by one
- Feature set comparison: Compare predefined feature sets
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm
from xgboost import XGBClassifier

# Predefined feature sets for comparison
FEATURE_SETS = {
    # Original 9 features from video paper
    'original_9d': [
        'mean', 'variance', 'skewness', 'kurtosis',
        'q25', 'q50', 'q75',
        'variance_mean_ratio', 'kurtosis_variance_ratio',
    ],
    # Recommended 7-feature universal set
    'universal_7': [
        'peak_to_peak', 'entropy', 'std', 'variance',
        'q95', 'iqr', 'coefficient_of_variation',
    ],
    # Top 5 by typical AUC
    'top_5_auc': [
        'peak_to_peak', 'entropy', 'std', 'variance', 'q95',
    ],
    # Spread measures only
    'spread_only': [
        'std', 'variance', 'peak_to_peak', 'iqr', 'coefficient_of_variation',
    ],
    # Central tendency only
    'central_only': [
        'mean', 'trimmed_mean', 'q50', 'median_mean_diff',
    ],
    # Distribution shape only
    'shape_only': [
        'skewness', 'kurtosis', 'bimodality_coefficient', 'entropy',
    ],
    # Percentiles only
    'percentiles_only': [
        'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95',
    ],
}


def _get_classifier(random_state: int = 42) -> XGBClassifier:
    """Get baseline XGBoost classifier."""
    return XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_lambda=1.0,
        n_estimators=100,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state,
        verbosity=0,
    )


def _cv_score(
    X: pd.DataFrame,
    y: np.ndarray,
    features: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    scoring: str = 'f1',
) -> float:
    """Get cross-validation score for a feature set.

    Args:
        X: Full feature DataFrame
        y: Labels
        features: Features to use
        cv_folds: Number of CV folds
        random_state: Random seed
        scoring: Scoring metric

    Returns:
        Mean CV score
    """
    if not features:
        return 0.0

    # Filter to available features
    available = [f for f in features if f in X.columns]
    if not available:
        return 0.0

    X_subset = X[available]
    clf = _get_classifier(random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X_subset, y, cv=cv, scoring=scoring)
    return float(np.mean(scores))


def leave_one_out_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    features: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-one-out ablation: measure impact of removing each feature.

    Args:
        X: Feature DataFrame
        y: Labels
        features: Base feature set
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print progress

    Returns:
        DataFrame with feature, full_score, without_score, drop columns
    """
    # Baseline with all features
    baseline_score = _cv_score(X, y, features, cv_folds, random_state)

    results = []
    iterator = tqdm(features, desc="Leave-one-out") if verbose else features

    for feature in iterator:
        remaining = [f for f in features if f != feature]
        score_without = _cv_score(X, y, remaining, cv_folds, random_state)
        drop = baseline_score - score_without

        results.append({
            'feature': feature,
            'full_score': baseline_score,
            'without_score': score_without,
            'drop': drop,
            'drop_pct': drop / baseline_score * 100 if baseline_score > 0 else 0,
        })

    df = pd.DataFrame(results)
    return df.sort_values('drop', ascending=False)


def incremental_addition_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    ranked_features: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Incremental addition: add features one by one in ranked order.

    Args:
        X: Feature DataFrame
        y: Labels
        ranked_features: Features in order of importance
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print progress

    Returns:
        DataFrame with n_features, features, score columns
    """
    results = []
    current_features = []

    iterator = enumerate(ranked_features, 1)
    if verbose:
        iterator = tqdm(list(iterator), desc="Incremental addition")

    for i, feature in iterator:
        current_features.append(feature)
        score = _cv_score(X, y, current_features, cv_folds, random_state)

        results.append({
            'n_features': i,
            'feature_added': feature,
            'features': current_features.copy(),
            'cv_f1': score,
        })

    return pd.DataFrame(results)


def feature_set_comparison(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_sets: Optional[Dict[str, List[str]]] = None,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare predefined feature sets.

    Args:
        X: Feature DataFrame
        y: Labels
        feature_sets: Dictionary of named feature sets (default: FEATURE_SETS)
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print progress

    Returns:
        DataFrame with set_name, n_features, cv_f1 columns
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS

    # Add "all features" set
    all_features = [f for f in X.columns if f != 'n_samples']
    feature_sets = {'all_features': all_features, **feature_sets}

    results = []
    iterator = feature_sets.items()
    if verbose:
        iterator = tqdm(list(iterator), desc="Feature sets")

    for set_name, features in iterator:
        # Filter to available features
        available = [f for f in features if f in X.columns]
        score = _cv_score(X, y, available, cv_folds, random_state)

        results.append({
            'set_name': set_name,
            'n_features': len(available),
            'features': available,
            'cv_f1': score,
        })

    df = pd.DataFrame(results)
    return df.sort_values('cv_f1', ascending=False)


def feature_group_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Ablation by removing entire feature groups.

    Args:
        X: Feature DataFrame
        y: Labels
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print progress

    Returns:
        DataFrame showing impact of removing each group
    """
    # Define feature groups
    feature_groups = {
        'basic_stats': ['mean', 'std', 'variance', 'min', 'max', 'peak_to_peak'],
        'distribution_shape': ['skewness', 'kurtosis', 'bimodality_coefficient'],
        'percentiles': ['q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr'],
        'tail_outliers': ['tail_weight_ratio', 'trimmed_mean'],
        'information': ['entropy', 'gini_coefficient', 'coefficient_of_variation'],
        'derived_ratios': [
            'variance_mean_ratio', 'kurtosis_variance_ratio',
            'skewness_kurtosis_ratio', 'iqr_range_ratio', 'median_mean_diff'
        ],
        'normality_tests': ['shapiro_p', 'normaltest_p'],
    }

    all_features = [f for f in X.columns if f != 'n_samples']
    baseline_score = _cv_score(X, y, all_features, cv_folds, random_state)

    results = []
    iterator = feature_groups.items()
    if verbose:
        iterator = tqdm(list(iterator), desc="Feature groups")

    for group_name, group_features in iterator:
        # Remove this group
        remaining = [f for f in all_features if f not in group_features]
        score_without = _cv_score(X, y, remaining, cv_folds, random_state)
        drop = baseline_score - score_without

        results.append({
            'group': group_name,
            'n_features_removed': len([f for f in group_features if f in X.columns]),
            'baseline_score': baseline_score,
            'without_score': score_without,
            'drop': drop,
            'drop_pct': drop / baseline_score * 100 if baseline_score > 0 else 0,
        })

    df = pd.DataFrame(results)
    return df.sort_values('drop', ascending=False)


class AblationStudy:
    """Run comprehensive ablation studies."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """Initialize ablation study.

        Args:
            X: Feature DataFrame
            y: Labels
            cv_folds: Number of CV folds
            random_state: Random seed
            verbose: Print progress
        """
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

        self.leave_one_out_results_: Optional[pd.DataFrame] = None
        self.incremental_results_: Optional[pd.DataFrame] = None
        self.feature_set_results_: Optional[pd.DataFrame] = None
        self.group_ablation_results_: Optional[pd.DataFrame] = None

    def run_all(
        self,
        base_features: Optional[List[str]] = None,
        ranked_features: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run all ablation studies.

        Args:
            base_features: Features for leave-one-out (default: all)
            ranked_features: Features in importance order for incremental

        Returns:
            Dictionary of result DataFrames
        """
        if base_features is None:
            base_features = [f for f in self.X.columns if f != 'n_samples']

        if ranked_features is None:
            ranked_features = base_features

        # Leave-one-out
        if self.verbose:
            print("\n=== Leave-One-Out Ablation ===")
        self.leave_one_out_results_ = leave_one_out_ablation(
            self.X, self.y, base_features,
            self.cv_folds, self.random_state, self.verbose
        )

        # Incremental addition
        if self.verbose:
            print("\n=== Incremental Addition ===")
        self.incremental_results_ = incremental_addition_ablation(
            self.X, self.y, ranked_features,
            self.cv_folds, self.random_state, self.verbose
        )

        # Feature set comparison
        if self.verbose:
            print("\n=== Feature Set Comparison ===")
        self.feature_set_results_ = feature_set_comparison(
            self.X, self.y, None,
            self.cv_folds, self.random_state, self.verbose
        )

        # Group ablation
        if self.verbose:
            print("\n=== Feature Group Ablation ===")
        self.group_ablation_results_ = feature_group_ablation(
            self.X, self.y,
            self.cv_folds, self.random_state, self.verbose
        )

        return {
            'leave_one_out': self.leave_one_out_results_,
            'incremental': self.incremental_results_,
            'feature_sets': self.feature_set_results_,
            'group_ablation': self.group_ablation_results_,
        }

    def plot_incremental(
        self,
        save_path: Optional[Path] = None,
        title: str = "Feature Addition Curve",
    ) -> None:
        """Plot incremental feature addition curve.

        Args:
            save_path: Path to save figure
            title: Plot title
        """
        if self.incremental_results_ is None:
            raise ValueError("Run run_all() first")

        df = self.incremental_results_

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['n_features'], df['cv_f1'], 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('CV F1 Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Mark the knee point (diminishing returns)
        best_idx = df['cv_f1'].idxmax()
        best_n = df.loc[best_idx, 'n_features']
        best_score = df.loc[best_idx, 'cv_f1']
        ax.axvline(x=best_n, color='r', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {best_n} features\nF1={best_score:.3f}',
                    xy=(best_n, best_score),
                    xytext=(best_n + 2, best_score - 0.02),
                    fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_feature_sets(
        self,
        save_path: Optional[Path] = None,
        title: str = "Feature Set Comparison",
    ) -> None:
        """Plot feature set comparison bar chart.

        Args:
            save_path: Path to save figure
            title: Plot title
        """
        if self.feature_set_results_ is None:
            raise ValueError("Run run_all() first")

        df = self.feature_set_results_.sort_values('cv_f1', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(df))

        colors = ['#2166ac' if name != 'all_features' else '#b2182b'
                  for name in df['set_name']]

        ax.barh(y_pos, df['cv_f1'], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name} ({n})" for name, n in
                           zip(df['set_name'], df['n_features'])])
        ax.set_xlabel('CV F1 Score')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_results(self, output_dir: Path) -> None:
        """Save all results to CSV files.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.leave_one_out_results_ is not None:
            self.leave_one_out_results_.to_csv(
                output_dir / 'leave_one_out.csv', index=False
            )

        if self.incremental_results_ is not None:
            # Don't save the features list column
            df = self.incremental_results_.drop(columns=['features'])
            df.to_csv(output_dir / 'incremental_addition.csv', index=False)

        if self.feature_set_results_ is not None:
            df = self.feature_set_results_.drop(columns=['features'])
            df.to_csv(output_dir / 'feature_set_comparison.csv', index=False)

        if self.group_ablation_results_ is not None:
            self.group_ablation_results_.to_csv(
                output_dir / 'group_ablation.csv', index=False
            )

"""Feature selection for classification training.

Implements greedy forward selection with cross-validation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def compute_feature_stability(
    X: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute domain stability scores for features using KS-test.

    Features with similar distributions across random splits are more likely
    to generalize to unseen domains.

    Args:
        X: Feature DataFrame
        n_splits: Number of random splits to average
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping feature name to stability score (0-1, higher = more stable)
    """
    rng = np.random.RandomState(random_state)
    stability_scores = {}

    for feature in X.columns:
        values = X[feature].values

        # Skip if all values are the same
        if np.std(values) < 1e-10:
            stability_scores[feature] = 1.0
            continue

        ks_stats = []
        for _ in range(n_splits):
            # Split into two random halves
            idx = rng.permutation(len(X))
            half1 = values[idx[:len(idx)//2]]
            half2 = values[idx[len(idx)//2:]]

            try:
                stat, _ = ks_2samp(half1, half2)
                ks_stats.append(stat)
            except Exception:
                ks_stats.append(0.0)

        # Lower KS stat = more stable distribution
        # Convert to stability score (0-1, higher = better)
        stability_scores[feature] = 1.0 - np.mean(ks_stats)

    return stability_scores


class GreedyFeatureSelector:
    """Greedy forward feature selection with cross-validation.

    Algorithm:
    1. Rank features by individual AUC-ROC
    2. Start with best single feature
    3. Greedily add features that improve CV F1 score by > threshold
    4. Stop at max_features or when no improvement
    5. If below min_features, add best remaining features by AUC ranking
    """

    def __init__(
        self,
        max_features: int = 15,
        min_features: int = 8,
        improvement_threshold: float = 0.005,  # 0.5% improvement required
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True,
        stability_weight: float = 0.0,  # v2.2: Weight for domain stability scoring
    ):
        """Initialize the feature selector.

        Args:
            max_features: Maximum number of features to select
            min_features: Minimum number of features to force (for robustness)
            improvement_threshold: Minimum F1 improvement to add a feature
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
            stability_weight: Weight for domain stability in ranking (0-1).
                              0 = pure AUC ranking, 1 = pure stability ranking.
                              Recommended: 0.3 for balanced selection.
        """
        self.max_features = max_features
        self.min_features = min_features
        self.improvement_threshold = improvement_threshold
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self.stability_weight = stability_weight

        self.selected_features_: List[str] = []
        self.feature_rankings_: Dict[str, float] = {}
        self.stability_scores_: Dict[str, float] = {}
        self.selection_history_: List[Dict] = []

    def _get_classifier(self) -> XGBClassifier:
        """Get XGBoost classifier with baseline hyperparameters."""
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
            random_state=self.random_state,
            verbosity=0,
        )

    def rank_features_by_auc(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Rank features by individual AUC-ROC, optionally weighted by stability.

        If stability_weight > 0, combines AUC and domain stability scores:
            combined = (1 - stability_weight) * auc + stability_weight * stability

        Args:
            X: Feature DataFrame
            y: Label array

        Returns:
            Dictionary mapping feature name to combined score
        """
        auc_rankings = {}

        for feature in X.columns:
            values = X[feature].values

            # Skip if all values are the same
            if np.std(values) < 1e-10:
                auc_rankings[feature] = 0.5
                continue

            try:
                auc = roc_auc_score(y, values)
                # Use max(auc, 1-auc) since direction doesn't matter
                auc = max(auc, 1 - auc)
                auc_rankings[feature] = auc
            except Exception:
                auc_rankings[feature] = 0.5

        # Compute stability scores if using stability weighting
        if self.stability_weight > 0:
            if self.verbose:
                print("\nComputing domain stability scores (KS-test)...")
            self.stability_scores_ = compute_feature_stability(
                X, n_splits=5, random_state=self.random_state
            )

            # Combine AUC and stability scores
            combined_rankings = {}
            for feature in X.columns:
                auc_score = auc_rankings.get(feature, 0.5)
                stability_score = self.stability_scores_.get(feature, 0.5)
                combined = (
                    (1 - self.stability_weight) * auc_score +
                    self.stability_weight * stability_score
                )
                combined_rankings[feature] = combined

            # Sort by combined score descending
            self.feature_rankings_ = dict(
                sorted(combined_rankings.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            # Pure AUC ranking
            self.feature_rankings_ = dict(
                sorted(auc_rankings.items(), key=lambda x: x[1], reverse=True)
            )

        return self.feature_rankings_

    def _evaluate_feature_set(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str],
    ) -> float:
        """Evaluate a feature set using cross-validation F1 score.

        Args:
            X: Full feature DataFrame
            y: Label array
            features: List of feature names to use

        Returns:
            Mean CV F1 score
        """
        if not features:
            return 0.0

        X_subset = X[features]
        clf = self._get_classifier()
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_val_score(clf, X_subset, y, cv=cv, scoring='f1')
        return float(np.mean(scores))

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        candidate_features: Optional[List[str]] = None,
    ) -> 'GreedyFeatureSelector':
        """Run greedy forward feature selection.

        Args:
            X: Feature DataFrame
            y: Label array
            candidate_features: Features to consider (default: all columns)

        Returns:
            self
        """
        if candidate_features is None:
            candidate_features = list(X.columns)

        # Filter to features that exist in X
        candidate_features = [f for f in candidate_features if f in X.columns]

        if self.verbose:
            print(f"Starting feature selection with {len(candidate_features)} candidates")

        # Step 1: Rank features by individual AUC
        self.rank_features_by_auc(X, y)

        if self.verbose:
            print("\nTop 10 features by individual AUC:")
            for i, (feat, auc) in enumerate(list(self.feature_rankings_.items())[:10]):
                print(f"  {i+1}. {feat}: {auc:.4f}")

        # Step 2: Start with best single feature
        ranked_features = [
            f for f in self.feature_rankings_.keys()
            if f in candidate_features
        ]

        if not ranked_features:
            raise ValueError("No valid candidate features found")

        best_feature = ranked_features[0]
        self.selected_features_ = [best_feature]
        best_score = self._evaluate_feature_set(X, y, self.selected_features_)

        self.selection_history_ = [{
            'step': 1,
            'feature_added': best_feature,
            'features': self.selected_features_.copy(),
            'cv_f1': best_score,
            'improvement': None,
        }]

        if self.verbose:
            print(f"\nStep 1: Added '{best_feature}' - CV F1: {best_score:.4f}")

        # Step 3: Greedily add features
        remaining_features = [f for f in ranked_features if f != best_feature]

        for step in range(2, self.max_features + 1):
            if not remaining_features:
                if self.verbose:
                    print("No more candidate features")
                break

            # Find best feature to add
            best_candidate = None
            best_new_score = best_score

            for candidate in remaining_features:
                test_features = self.selected_features_ + [candidate]
                score = self._evaluate_feature_set(X, y, test_features)

                if score > best_new_score:
                    best_new_score = score
                    best_candidate = candidate

            # Check if improvement exceeds threshold
            improvement = best_new_score - best_score

            if best_candidate is not None and improvement > self.improvement_threshold:
                self.selected_features_.append(best_candidate)
                remaining_features.remove(best_candidate)
                best_score = best_new_score

                self.selection_history_.append({
                    'step': step,
                    'feature_added': best_candidate,
                    'features': self.selected_features_.copy(),
                    'cv_f1': best_score,
                    'improvement': improvement,
                })

                if self.verbose:
                    print(f"Step {step}: Added '{best_candidate}' - "
                          f"CV F1: {best_score:.4f} (+{improvement:.4f})")
            else:
                if self.verbose:
                    print(f"\nStopping greedy selection: No feature improves F1 by > {self.improvement_threshold:.3f}")
                break

        # Step 4: Enforce minimum features if below threshold
        if len(self.selected_features_) < self.min_features and remaining_features:
            if self.verbose:
                print(f"\nAdding features to reach min_features={self.min_features}:")

            while len(self.selected_features_) < self.min_features and remaining_features:
                # Add next best feature by AUC ranking
                for feat in ranked_features:
                    if feat not in self.selected_features_ and feat in remaining_features:
                        self.selected_features_.append(feat)
                        remaining_features.remove(feat)

                        # Evaluate new feature set
                        new_score = self._evaluate_feature_set(X, y, self.selected_features_)

                        self.selection_history_.append({
                            'step': len(self.selected_features_),
                            'feature_added': feat,
                            'features': self.selected_features_.copy(),
                            'cv_f1': new_score,
                            'improvement': None,
                            'reason': 'min_features',
                        })

                        if self.verbose:
                            print(f"  Added '{feat}' (AUC: {self.feature_rankings_.get(feat, 0):.4f}) - CV F1: {new_score:.4f}")
                        break

        if self.verbose:
            print(f"\nSelected {len(self.selected_features_)} features:")
            for i, feat in enumerate(self.selected_features_):
                print(f"  {i+1}. {feat}")

        return self

    def get_selected_features(self) -> List[str]:
        """Get the list of selected features.

        Returns:
            List of selected feature names
        """
        return self.selected_features_.copy()

    def get_selection_summary(self) -> pd.DataFrame:
        """Get summary of feature selection process.

        Returns:
            DataFrame with selection history
        """
        return pd.DataFrame(self.selection_history_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X to include only selected features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with only selected features
        """
        if not self.selected_features_:
            raise ValueError("No features selected. Call fit() first.")
        return X[self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        candidate_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Feature DataFrame
            y: Label array
            candidate_features: Features to consider

        Returns:
            DataFrame with only selected features
        """
        self.fit(X, y, candidate_features)
        return self.transform(X)


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    max_features: int = 15,
    min_features: int = 8,
    improvement_threshold: float = 0.005,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    stability_weight: float = 0.0,
) -> Tuple[List[str], pd.DataFrame]:
    """Convenience function for feature selection.

    Args:
        X: Feature DataFrame
        y: Label array
        max_features: Maximum features to select
        min_features: Minimum features to force (for robustness)
        improvement_threshold: Minimum F1 improvement required
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print progress
        stability_weight: Weight for domain stability (0-1). 0.3 recommended for
                          better generalization.

    Returns:
        Tuple of (selected_features, selection_history_df)
    """
    selector = GreedyFeatureSelector(
        max_features=max_features,
        min_features=min_features,
        improvement_threshold=improvement_threshold,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=verbose,
        stability_weight=stability_weight,
    )

    selector.fit(X, y)

    return selector.get_selected_features(), selector.get_selection_summary()

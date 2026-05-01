"""XGBoost training with cross-validation.

Handles training, evaluation, and model persistence.
"""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

from .feature_selector import GreedyFeatureSelector


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # XGBoost hyperparameters
    learning_rate: float = 0.1
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_lambda: float = 1.0  # L2 regularization
    reg_alpha: float = 0.0   # L1 regularization (v2.2+)
    n_estimators: int = 100
    early_stopping_rounds: int = 10

    # Training settings
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

    # Feature selection
    max_features: int = 15
    improvement_threshold: float = 0.005
    stability_weight: float = 0.0  # v2.2+: Weight for domain stability (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# v2.2 config with stronger regularization for better generalization
# Based on SAM research: flatter loss landscape = better OOD performance
V2_2_CONFIG = TrainingConfig(
    learning_rate=0.05,      # Lower = flatter landscape
    max_depth=4,             # Shallower trees = less overfitting
    min_child_weight=3,      # More samples per leaf
    n_estimators=200,        # More weak learners
    reg_lambda=5.0,          # Stronger L2
    reg_alpha=0.5,           # Add L1 sparsity
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    stability_weight=0.3,    # Weight for domain stability in feature selection
)


@dataclass
class TrainingResult:
    """Result of training a model."""
    config_name: str
    audio_type: str
    embedding_model: str

    # Dataset info
    n_samples: int
    n_train: int
    n_test: int
    n_real: int
    n_fake: int

    # Feature selection
    selected_features: List[str]
    feature_rankings: Dict[str, float]

    # CV results (on training set)
    cv_scores: Dict[str, List[float]]
    cv_mean_scores: Dict[str, float]

    # Test set results (held-out)
    test_scores: Dict[str, float]
    confusion_matrix: List[List[int]]

    # Model info
    feature_importance: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_name': self.config_name,
            'audio_type': self.audio_type,
            'embedding_model': self.embedding_model,
            'n_samples': self.n_samples,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_real': self.n_real,
            'n_fake': self.n_fake,
            'selected_features': self.selected_features,
            'feature_rankings': self.feature_rankings,
            'cv_scores': self.cv_scores,
            'cv_mean_scores': self.cv_mean_scores,
            'test_scores': self.test_scores,
            'confusion_matrix': self.confusion_matrix,
            'feature_importance': self.feature_importance,
        }


class Trainer:
    """XGBoost trainer with cross-validation and feature selection."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model_: Optional[XGBClassifier] = None
        self.feature_selector_: Optional[GreedyFeatureSelector] = None
        self.result_: Optional[TrainingResult] = None
        self.scaler_ = None  # RobustScaler for v2+ models

    def _create_classifier(self) -> XGBClassifier:
        """Create XGBoost classifier with config hyperparameters."""
        return XGBClassifier(
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,  # v2.2+: L1 regularization
            n_estimators=self.config.n_estimators,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.config.random_state,
            verbosity=0,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        config_name: str,
        audio_type: str,
        embedding_model: str,
        stratify_groups: Optional[np.ndarray] = None,
        run_feature_selection: bool = True,
        features: Optional[List[str]] = None,
        verbose: bool = True,
        scaler=None,
    ) -> TrainingResult:
        """Train model with cross-validation and feature selection.

        Args:
            X: Feature DataFrame
            y: Label array
            config_name: Name for this training configuration
            audio_type: Audio type being trained
            embedding_model: Embedding model used
            stratify_groups: Groups for stratified split (for universal model)
            run_feature_selection: Whether to run feature selection
            features: Pre-selected features (if not running selection)
            verbose: Print progress
            scaler: Pre-fitted RobustScaler (for v2+ models with normalization)

        Returns:
            TrainingResult with all metrics and info
        """
        # Store scaler for saving with model
        self.scaler_ = scaler

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {config_name}")
            print(f"Audio type: {audio_type}, Embedding: {embedding_model}")
            print(f"{'='*60}")

        # Dataset stats
        n_samples = len(y)
        n_real = int(np.sum(y == 0))
        n_fake = int(np.sum(y == 1))

        if verbose:
            print(f"\nDataset: {n_samples} samples (real: {n_real}, fake: {n_fake})")

        # Split into train/test
        stratify = stratify_groups if stratify_groups is not None else y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=stratify,
            random_state=self.config.random_state,
        )

        n_train = len(y_train)
        n_test = len(y_test)

        if verbose:
            print(f"Train: {n_train}, Test: {n_test}")

        # Feature selection on training set only
        if run_feature_selection:
            if verbose:
                print("\n--- Feature Selection ---")
                if self.config.stability_weight > 0:
                    print(f"Using domain stability weighting: {self.config.stability_weight:.1%}")

            self.feature_selector_ = GreedyFeatureSelector(
                max_features=self.config.max_features,
                improvement_threshold=self.config.improvement_threshold,
                cv_folds=self.config.cv_folds,
                random_state=self.config.random_state,
                verbose=verbose,
                stability_weight=self.config.stability_weight,
            )
            self.feature_selector_.fit(X_train, y_train)
            selected_features = self.feature_selector_.get_selected_features()
            feature_rankings = self.feature_selector_.feature_rankings_
        else:
            selected_features = features or list(X.columns)
            feature_rankings = {}

        # Use only selected features
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]

        # Cross-validation on training set
        if verbose:
            print("\n--- Cross-Validation ---")

        clf = self._create_classifier()
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(
            clf, X_train_sel, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

        cv_scores = {
            metric: cv_results[f'test_{metric}'].tolist()
            for metric in scoring
        }
        cv_mean_scores = {
            metric: float(np.mean(scores))
            for metric, scores in cv_scores.items()
        }

        if verbose:
            print("CV Results (mean ± std):")
            for metric, scores in cv_scores.items():
                print(f"  {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # Train final model on full training set
        if verbose:
            print("\n--- Final Model Training ---")

        self.model_ = self._create_classifier()
        self.model_.fit(X_train_sel, y_train)

        # Evaluate on held-out test set
        if verbose:
            print("\n--- Test Set Evaluation ---")

        y_pred = self.model_.predict(X_test_sel)
        y_proba = self.model_.predict_proba(X_test_sel)[:, 1]

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.metrics import (
            confusion_matrix as compute_cm,
        )

        test_scores = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
        }

        # Per-class metrics
        test_scores['precision_real'] = float(precision_score(y_test, y_pred, pos_label=0))
        test_scores['recall_real'] = float(recall_score(y_test, y_pred, pos_label=0))
        test_scores['f1_real'] = float(f1_score(y_test, y_pred, pos_label=0))
        test_scores['precision_fake'] = float(precision_score(y_test, y_pred, pos_label=1))
        test_scores['recall_fake'] = float(recall_score(y_test, y_pred, pos_label=1))
        test_scores['f1_fake'] = float(f1_score(y_test, y_pred, pos_label=1))

        # Class balance ratio
        test_scores['class_balance_ratio'] = min(
            test_scores['f1_real'], test_scores['f1_fake']
        ) / max(test_scores['f1_real'], test_scores['f1_fake'])

        cm = compute_cm(y_test, y_pred).tolist()

        if verbose:
            print("Test Results:")
            for metric, score in test_scores.items():
                print(f"  {metric}: {score:.4f}")
            print("\nConfusion Matrix:")
            print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
            print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")

        # Feature importance
        feature_importance = dict(zip(
            selected_features,
            self.model_.feature_importances_.tolist(),
        ))

        # Create result
        self.result_ = TrainingResult(
            config_name=config_name,
            audio_type=audio_type,
            embedding_model=embedding_model,
            n_samples=n_samples,
            n_train=n_train,
            n_test=n_test,
            n_real=n_real,
            n_fake=n_fake,
            selected_features=selected_features,
            feature_rankings=feature_rankings,
            cv_scores=cv_scores,
            cv_mean_scores=cv_mean_scores,
            test_scores=test_scores,
            confusion_matrix=cm,
            feature_importance=feature_importance,
        )

        return self.result_

    def save_model(self, path: Path) -> None:
        """Save trained model to file.

        Args:
            path: Path to save model
        """
        if self.model_ is None:
            raise ValueError("No model trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model_,
                'features': self.result_.selected_features if self.result_ else [],
                'config': self.config.to_dict(),
                'scaler': self.scaler_,  # v2+: RobustScaler for normalization
            }, f)

    def load_model(self, path: Path) -> None:
        """Load trained model from file.

        Args:
            path: Path to model file
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model_ = data['model']
        self.scaler_ = data.get('scaler', None)  # v2+: Load scaler if present

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature DataFrame

        Returns:
            Binary predictions
        """
        if self.model_ is None:
            raise ValueError("No model trained. Call train() first.")

        features = self.result_.selected_features if self.result_ else list(X.columns)
        return self.model_.predict(X[features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Probability of fake class
        """
        if self.model_ is None:
            raise ValueError("No model trained. Call train() first.")

        features = self.result_.selected_features if self.result_ else list(X.columns)
        return self.model_.predict_proba(X[features])[:, 1]

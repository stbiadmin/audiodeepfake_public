"""XGBoost classifier wrapper for audio deepfake detection.

Provides a consistent interface for training and inference.
"""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass
class XGBoostHyperparameters:
    """Hyperparameters for XGBoost classifier.

    Default values based on reference paper and best practices.
    """
    learning_rate: float = 0.1
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_lambda: float = 1.0
    n_estimators: int = 100
    early_stopping_rounds: int = 10
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioDeepfakeClassifier:
    """XGBoost classifier for audio deepfake detection.

    Wraps XGBClassifier with methods specific to our use case.
    """

    def __init__(
        self,
        hyperparams: Optional[XGBoostHyperparameters] = None,
        features: Optional[List[str]] = None,
    ):
        """Initialize classifier.

        Args:
            hyperparams: XGBoost hyperparameters
            features: List of feature names to use (if None, use all)
        """
        self.hyperparams = hyperparams or XGBoostHyperparameters()
        self.features = features
        self.model_: Optional[XGBClassifier] = None
        self.is_fitted_ = False

    def _create_model(self) -> XGBClassifier:
        """Create XGBClassifier with configured hyperparameters."""
        return XGBClassifier(
            learning_rate=self.hyperparams.learning_rate,
            max_depth=self.hyperparams.max_depth,
            min_child_weight=self.hyperparams.min_child_weight,
            subsample=self.hyperparams.subsample,
            colsample_bytree=self.hyperparams.colsample_bytree,
            gamma=self.hyperparams.gamma,
            reg_lambda=self.hyperparams.reg_lambda,
            n_estimators=self.hyperparams.n_estimators,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.hyperparams.random_state,
            verbosity=0,
        )

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features if feature list is specified.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with selected features
        """
        if self.features is not None:
            return X[self.features]
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
        early_stopping: bool = False,
    ) -> 'AudioDeepfakeClassifier':
        """Train the classifier.

        Args:
            X: Feature DataFrame
            y: Label array (0=real, 1=fake)
            eval_set: Optional validation set for early stopping
            early_stopping: Whether to use early stopping

        Returns:
            self
        """
        X_train = self._select_features(X)
        self.model_ = self._create_model()

        fit_params = {}

        if early_stopping and eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._select_features(X_val)
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        self.model_.fit(X_train, y, **fit_params)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Binary predictions (0=real, 1=fake)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        X_pred = self._select_features(X)
        return self.model_.predict(X_pred)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with [p_real, p_fake]
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        X_pred = self._select_features(X)
        return self.model_.predict_proba(X_pred)

    def get_feature_importance(
        self,
        importance_type: str = 'gain',
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get importance from booster
        booster = self.model_.get_booster()
        importance = booster.get_score(importance_type=importance_type)

        # Map feature indices to names
        if self.features is not None:
            feature_names = self.features
        else:
            feature_names = [f'f{i}' for i in range(self.model_.n_features_in_)]

        # Convert to named dict
        result = {}
        for fname in feature_names:
            # XGBoost uses f0, f1, etc. by default
            idx = feature_names.index(fname)
            key = f'f{idx}'
            result[fname] = importance.get(key, 0.0)

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_sklearn_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from sklearn interface.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = self.model_.feature_importances_

        if self.features is not None:
            feature_names = self.features
        else:
            feature_names = [f'f{i}' for i in range(len(importances))]

        result = dict(zip(feature_names, importances.tolist()))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save model to file.

        Args:
            path: Path to save model
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model_,
                'hyperparams': self.hyperparams.to_dict(),
                'features': self.features,
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'AudioDeepfakeClassifier':
        """Load model from file.

        Args:
            path: Path to model file

        Returns:
            Loaded classifier
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        hyperparams = XGBoostHyperparameters(**data['hyperparams'])
        classifier = cls(hyperparams=hyperparams, features=data['features'])
        classifier.model_ = data['model']
        classifier.is_fitted_ = True

        return classifier

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        n_features = len(self.features) if self.features else "all"
        return f"AudioDeepfakeClassifier({status}, features={n_features})"

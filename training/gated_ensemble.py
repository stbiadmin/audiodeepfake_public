"""Gated Ensemble Classifier for Audio Deepfake Detection.

Implements dynamic sample routing based on DoGEN (Domain Gating Ensemble Networks)
and DEER research to handle domain shift between training and in-the-wild data.

Core idea: Instead of one model for all samples, train multiple domain-specific
experts and a gating network that routes each sample to the best expert(s).

References:
- DoGEN: https://arxiv.org/html/2505.13855v1
- DEER: https://arxiv.org/html/2511.01192
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loader import ALL_FEATURES


@dataclass
class GatedEnsembleConfig:
    """Configuration for a gated ensemble."""
    name: str
    model_names: List[str]
    gating_method: str = 'learned'  # 'learned', 'rule_based', 'feature_profile'
    top_k: int = 2  # Number of experts to use per sample (for top-k routing)
    temperature: float = 1.0  # Softmax temperature for gating

    # For rule-based routing
    routing_rules: Optional[Dict[str, Any]] = None

    # For learned gating
    gating_features: Optional[List[str]] = None  # Features to use for gating (None = all)

    def __post_init__(self):
        if self.gating_features is None:
            # Default: use all features for gating
            self.gating_features = [f for f in ALL_FEATURES
                                   if f not in ['shapiro_p', 'normaltest_p']]


class GatingNetwork:
    """Neural network for learning sample-to-expert routing.

    Takes sample features as input and outputs expert weights.
    """

    def __init__(
        self,
        n_experts: int,
        input_features: List[str],
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        """Initialize gating network.

        Args:
            n_experts: Number of expert models to route to
            input_features: Feature names to use as input
            hidden_dim: Hidden layer dimension
            temperature: Softmax temperature (higher = more uniform)
        """
        self.n_experts = n_experts
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # MLP for gating: features -> expert weights
        self.network = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )

        # Scaler for input features
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        expert_labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'GatingNetwork':
        """Train gating network to predict best expert.

        Args:
            X: Feature DataFrame
            expert_labels: Which expert is best for each sample (0 to n_experts-1)
            sample_weight: Optional sample weights

        Returns:
            Self
        """
        X_gating = X[self.input_features].values
        X_scaled = self.scaler.fit_transform(X_gating)

        self.network.fit(X_scaled, expert_labels)
        self.is_fitted = True

        return self

    def predict_weights(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expert weights for samples.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, n_experts) with expert weights
        """
        if not self.is_fitted:
            # Default to uniform weights if not fitted
            return np.ones((len(X), self.n_experts)) / self.n_experts

        X_gating = X[self.input_features].values
        X_scaled = self.scaler.transform(X_gating)

        # Get probability for each expert
        weights = self.network.predict_proba(X_scaled)

        # Apply temperature scaling
        if self.temperature != 1.0:
            log_weights = np.log(weights + 1e-10)
            weights = np.exp(log_weights / self.temperature)
            weights = weights / weights.sum(axis=1, keepdims=True)

        return weights

    def predict_top_k(self, X: pd.DataFrame, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Predict top-k experts for each sample.

        Args:
            X: Feature DataFrame
            k: Number of top experts

        Returns:
            Tuple of (expert_indices, expert_weights) arrays
        """
        weights = self.predict_weights(X)

        # Get top-k indices and their weights
        top_k_indices = np.argsort(weights, axis=1)[:, -k:][:, ::-1]
        top_k_weights = np.take_along_axis(weights, top_k_indices, axis=1)

        # Renormalize weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(axis=1, keepdims=True)

        return top_k_indices, top_k_weights


class RuleBasedRouter:
    """Simple rule-based routing based on feature thresholds.

    Useful when domain characteristics are known (e.g., ITW has high mean similarity).
    """

    def __init__(self, rules: Dict[str, Any]):
        """Initialize with routing rules.

        Args:
            rules: Dictionary defining routing rules. Example:
                {
                    'feature': 'mean',
                    'threshold': 0.85,
                    'above_threshold_expert': 0,  # Route high-mean samples to expert 0
                    'below_threshold_expert': 1,  # Route low-mean samples to expert 1
                }
        """
        self.rules = rules

    def route(self, X: pd.DataFrame) -> np.ndarray:
        """Determine routing for each sample.

        Args:
            X: Feature DataFrame

        Returns:
            Array of expert indices for each sample
        """
        feature = self.rules.get('feature', 'mean')
        threshold = self.rules.get('threshold', 0.85)
        above_expert = self.rules.get('above_threshold_expert', 0)
        below_expert = self.rules.get('below_threshold_expert', 1)

        values = X[feature].values
        routing = np.where(values > threshold, above_expert, below_expert)

        return routing


class FeatureProfileRouter:
    """Route based on how similar sample features are to training domains.

    Uses cosine similarity between sample features and domain centroids.
    """

    def __init__(self, n_experts: int):
        """Initialize router.

        Args:
            n_experts: Number of experts/domains
        """
        self.n_experts = n_experts
        self.domain_centroids: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        domain_labels: np.ndarray,
        features: Optional[List[str]] = None,
    ) -> 'FeatureProfileRouter':
        """Fit router by computing domain centroids.

        Args:
            X: Feature DataFrame
            domain_labels: Domain/expert label for each sample
            features: Features to use (None = all)

        Returns:
            Self
        """
        self.feature_names = features or list(X.columns)
        X_features = X[self.feature_names].values

        # Normalize features
        X_scaled = self.scaler.fit_transform(X_features)

        # Compute centroid for each domain
        self.domain_centroids = np.zeros((self.n_experts, len(self.feature_names)))
        for domain in range(self.n_experts):
            mask = domain_labels == domain
            if mask.any():
                self.domain_centroids[domain] = X_scaled[mask].mean(axis=0)

        self.is_fitted = True
        return self

    def route(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Route samples to nearest domain centroid.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (expert_indices, similarity_scores)
        """
        if not self.is_fitted:
            # Default to expert 0 if not fitted
            return np.zeros(len(X), dtype=int), np.ones(len(X))

        X_features = X[self.feature_names].values
        X_scaled = self.scaler.transform(X_features)

        # Compute cosine similarity to each centroid
        # Normalize both
        X_norm = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-10)
        centroids_norm = self.domain_centroids / (np.linalg.norm(self.domain_centroids, axis=1, keepdims=True) + 1e-10)

        similarities = X_norm @ centroids_norm.T  # (n_samples, n_experts)

        # Route to most similar domain
        expert_indices = np.argmax(similarities, axis=1)
        similarity_scores = np.max(similarities, axis=1)

        return expert_indices, similarity_scores


class GatedEnsembleClassifier:
    """Gated ensemble that routes samples to domain-specific experts.

    Extends the basic ensemble with intelligent routing based on:
    1. Learned gating network (DoGEN-style)
    2. Rule-based routing (threshold on features)
    3. Feature profile routing (nearest domain centroid)
    """

    def __init__(
        self,
        models_dir: Path,
        config: GatedEnsembleConfig,
        verbose: bool = True,
    ):
        """Initialize gated ensemble.

        Args:
            models_dir: Directory containing model pickle files
            config: Gated ensemble configuration
            verbose: Print progress
        """
        self.models_dir = Path(models_dir)
        self.config = config
        self.verbose = verbose

        self.models_: List[Dict[str, Any]] = []
        self.gating_network: Optional[GatingNetwork] = None
        self.rule_router: Optional[RuleBasedRouter] = None
        self.profile_router: Optional[FeatureProfileRouter] = None

        self._load_models()
        self._init_routing()

    def _load_models(self) -> None:
        """Load all expert models."""
        for model_name in self.config.model_names:
            model_path = self.models_dir / f"{model_name}.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            model_info = {
                'name': model_name,
                'model': data['model'],
                'features': data['features'],
                'scaler': data.get('scaler'),
            }

            self.models_.append(model_info)

            if self.verbose:
                print(f"Loaded expert: {model_name} ({len(data['features'])} features)")

    def _init_routing(self) -> None:
        """Initialize routing mechanism based on config."""
        n_experts = len(self.models_)

        if self.config.gating_method == 'learned':
            self.gating_network = GatingNetwork(
                n_experts=n_experts,
                input_features=self.config.gating_features,
                temperature=self.config.temperature,
            )

        elif self.config.gating_method == 'rule_based':
            if self.config.routing_rules is None:
                # Default rule: route high-similarity samples to expert 0
                self.config.routing_rules = {
                    'feature': 'mean',
                    'threshold': 0.85,
                    'above_threshold_expert': 0,
                    'below_threshold_expert': 1,
                }
            self.rule_router = RuleBasedRouter(self.config.routing_rules)

        elif self.config.gating_method == 'feature_profile':
            self.profile_router = FeatureProfileRouter(n_experts)

    def fit_gating(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
        method: str = 'oracle',
    ) -> 'GatedEnsembleClassifier':
        """Train the gating network.

        Args:
            X: Feature DataFrame with ALL_FEATURES
            y: True labels
            domain_labels: Domain labels (which training set each sample is from).
                          If None and method='oracle', determined by best expert.
            method: How to determine expert labels:
                - 'oracle': Use true labels to find best expert per sample
                - 'domain': Use provided domain_labels
                - 'confidence': Route uncertain samples to different experts

        Returns:
            Self
        """
        if self.config.gating_method == 'rule_based':
            if self.verbose:
                print("Rule-based routing doesn't need fitting.")
            return self

        if self.verbose:
            print(f"\nFitting gating network using '{method}' method...")

        if self.config.gating_method == 'feature_profile':
            if domain_labels is None:
                # Use oracle method to get domain labels
                domain_labels = self._get_oracle_expert_labels(X, y)
            self.profile_router.fit(X, domain_labels, self.config.gating_features)

        elif self.config.gating_method == 'learned':
            if method == 'oracle':
                expert_labels = self._get_oracle_expert_labels(X, y)
            elif method == 'domain':
                if domain_labels is None:
                    raise ValueError("domain_labels required for 'domain' method")
                expert_labels = domain_labels
            elif method == 'confidence':
                expert_labels = self._get_confidence_expert_labels(X, y)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.gating_network.fit(X, expert_labels)

            if self.verbose:
                # Report gating distribution
                unique, counts = np.unique(expert_labels, return_counts=True)
                print("Expert label distribution:")
                for expert, count in zip(unique, counts):
                    pct = count / len(expert_labels) * 100
                    print(f"  Expert {expert} ({self.config.model_names[expert]}): {count} ({pct:.1f}%)")

        return self

    def _get_oracle_expert_labels(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """Determine best expert for each sample using true labels (oracle).

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            Array of best expert indices
        """
        n_samples = len(X)
        n_experts = len(self.models_)

        # Get predictions from each expert
        expert_correct = np.zeros((n_samples, n_experts), dtype=bool)
        expert_confidence = np.zeros((n_samples, n_experts))

        for i, model_info in enumerate(self.models_):
            X_model = self._prepare_features(X, model_info)
            y_pred = model_info['model'].predict(X_model)
            y_proba = model_info['model'].predict_proba(X_model)[:, 1]

            expert_correct[:, i] = (y_pred == y)
            expert_confidence[:, i] = np.abs(y_proba - 0.5)  # Distance from decision boundary

        # For each sample, find best expert
        # Prefer correct experts; among correct, prefer higher confidence
        expert_labels = np.zeros(n_samples, dtype=int)

        for j in range(n_samples):
            correct_experts = np.where(expert_correct[j])[0]
            if len(correct_experts) > 0:
                # Pick most confident among correct
                best = correct_experts[np.argmax(expert_confidence[j, correct_experts])]
            else:
                # No expert is correct; pick most confident (closest to correct)
                best = np.argmax(expert_confidence[j])
            expert_labels[j] = best

        return expert_labels

    def _get_confidence_expert_labels(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """Route based on confidence: uncertain samples go to secondary expert.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            Expert labels
        """
        # Use first expert's confidence to route
        model_info = self.models_[0]
        X_model = self._prepare_features(X, model_info)
        y_proba = model_info['model'].predict_proba(X_model)[:, 1]

        # Uncertain samples (confidence < 0.7) go to expert 1
        confidence = np.abs(y_proba - 0.5) * 2  # Scale to [0, 1]
        expert_labels = np.where(confidence < 0.7, 1, 0)

        return expert_labels

    def _prepare_features(
        self,
        X: pd.DataFrame,
        model_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """Prepare features for a specific model.

        Args:
            X: Input DataFrame with ALL_FEATURES
            model_info: Model information dict

        Returns:
            Prepared feature DataFrame
        """
        if model_info['scaler'] is not None:
            X_scaled = model_info['scaler'].transform(X[ALL_FEATURES].values)
            X_scaled_df = pd.DataFrame(X_scaled, columns=ALL_FEATURES, index=X.index)
            return X_scaled_df[model_info['features']]
        else:
            return X[model_info['features']]

    def predict_proba(
        self,
        X: pd.DataFrame,
        return_routing: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get prediction probabilities with gated routing.

        Args:
            X: Feature DataFrame with ALL_FEATURES
            return_routing: If True, also return routing decisions

        Returns:
            Probabilities for fake class (and optionally routing info)
        """
        n_samples = len(X)
        n_experts = len(self.models_)

        # Get all expert predictions
        expert_probas = np.zeros((n_samples, n_experts))
        for i, model_info in enumerate(self.models_):
            X_model = self._prepare_features(X, model_info)
            expert_probas[:, i] = model_info['model'].predict_proba(X_model)[:, 1]

        # Get routing weights
        if self.config.gating_method == 'learned':
            if self.config.top_k < n_experts:
                # Top-k routing
                expert_indices, weights = self.gating_network.predict_top_k(X, self.config.top_k)
                # Compute weighted sum for top-k experts only
                final_probas = np.zeros(n_samples)
                for j in range(n_samples):
                    for k in range(self.config.top_k):
                        expert_idx = expert_indices[j, k]
                        final_probas[j] += weights[j, k] * expert_probas[j, expert_idx]
            else:
                # Full soft routing
                weights = self.gating_network.predict_weights(X)
                final_probas = np.sum(expert_probas * weights, axis=1)
                expert_indices = np.argmax(weights, axis=1)

        elif self.config.gating_method == 'rule_based':
            expert_indices = self.rule_router.route(X)
            # Hard routing
            final_probas = expert_probas[np.arange(n_samples), expert_indices]
            weights = None

        elif self.config.gating_method == 'feature_profile':
            expert_indices, similarity_scores = self.profile_router.route(X)
            # Hard routing
            final_probas = expert_probas[np.arange(n_samples), expert_indices]
            weights = None

        else:
            # Default: equal weights
            final_probas = expert_probas.mean(axis=1)
            expert_indices = np.zeros(n_samples, dtype=int)
            weights = None

        if return_routing:
            return final_probas, expert_indices
        return final_probas

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Get binary predictions.

        Args:
            X: Feature DataFrame
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Evaluate gated ensemble on a dataset.

        Args:
            X: Feature DataFrame
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix,
        )

        y_proba, routing = self.predict_proba(X, return_routing=True)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, y_proba)),
        }

        # Per-class metrics
        metrics['f1_real'] = float(f1_score(y, y_pred, pos_label=0, zero_division=0))
        metrics['f1_fake'] = float(f1_score(y, y_pred, pos_label=1, zero_division=0))

        # Routing statistics
        unique, counts = np.unique(routing, return_counts=True)
        for expert_idx, count in zip(unique, counts):
            expert_name = self.config.model_names[expert_idx]
            metrics[f'routed_to_{expert_name}'] = int(count)
            metrics[f'routed_to_{expert_name}_pct'] = float(count / len(y) * 100)

        # Per-expert accuracy (for routed samples)
        for expert_idx in range(len(self.models_)):
            mask = routing == expert_idx
            if mask.any():
                expert_acc = float(accuracy_score(y[mask], y_pred[mask]))
                metrics[f'accuracy_expert_{expert_idx}'] = expert_acc

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

        return metrics

    def save(self, path: Path) -> None:
        """Save gated ensemble to file.

        Args:
            path: Path to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'config': self.config,
            'model_names': self.config.model_names,
            'gating_method': self.config.gating_method,
        }

        if self.gating_network is not None and self.gating_network.is_fitted:
            data['gating_network'] = {
                'network': self.gating_network.network,
                'scaler': self.gating_network.scaler,
                'input_features': self.gating_network.input_features,
                'n_experts': self.gating_network.n_experts,
                'temperature': self.gating_network.temperature,
            }

        if self.rule_router is not None:
            data['routing_rules'] = self.rule_router.rules

        if self.profile_router is not None and self.profile_router.is_fitted:
            data['profile_router'] = {
                'domain_centroids': self.profile_router.domain_centroids,
                'feature_names': self.profile_router.feature_names,
                'scaler': self.profile_router.scaler,
            }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        if self.verbose:
            print(f"Saved gated ensemble to: {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        models_dir: Path,
        verbose: bool = True,
    ) -> 'GatedEnsembleClassifier':
        """Load gated ensemble from file.

        Args:
            path: Path to saved ensemble
            models_dir: Directory containing expert models
            verbose: Print progress

        Returns:
            Loaded GatedEnsembleClassifier
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Create ensemble
        ensemble = cls(models_dir, data['config'], verbose=verbose)

        # Restore gating network if present
        if 'gating_network' in data:
            gn_data = data['gating_network']
            ensemble.gating_network = GatingNetwork(
                n_experts=gn_data['n_experts'],
                input_features=gn_data['input_features'],
                temperature=gn_data['temperature'],
            )
            ensemble.gating_network.network = gn_data['network']
            ensemble.gating_network.scaler = gn_data['scaler']
            ensemble.gating_network.is_fitted = True

        # Restore rule router if present
        if 'routing_rules' in data:
            ensemble.rule_router = RuleBasedRouter(data['routing_rules'])

        # Restore profile router if present
        if 'profile_router' in data:
            pr_data = data['profile_router']
            ensemble.profile_router = FeatureProfileRouter(len(ensemble.models_))
            ensemble.profile_router.domain_centroids = pr_data['domain_centroids']
            ensemble.profile_router.feature_names = pr_data['feature_names']
            ensemble.profile_router.scaler = pr_data['scaler']
            ensemble.profile_router.is_fitted = True

        return ensemble


# Pre-defined gated ensemble configurations
GATED_ENSEMBLE_CONFIGS = {
    # Learned gating with DeepSpeak and SingleVoice experts
    'gated_ds_sv_learned': GatedEnsembleConfig(
        name='gated_ds_sv_learned',
        model_names=['ds_msclap_model', 'sv_msclap_model'],
        gating_method='learned',
        top_k=2,
        temperature=1.0,
    ),

    # Rule-based routing: high similarity → DS expert
    'gated_ds_sv_rule': GatedEnsembleConfig(
        name='gated_ds_sv_rule',
        model_names=['ds_msclap_model', 'sv_msclap_model'],
        gating_method='rule_based',
        routing_rules={
            'feature': 'mean',
            'threshold': 0.85,
            'above_threshold_expert': 0,  # DS for high-similarity (like ITW real)
            'below_threshold_expert': 1,  # SV for lower-similarity
        },
    ),

    # Feature profile routing
    'gated_ds_sv_profile': GatedEnsembleConfig(
        name='gated_ds_sv_profile',
        model_names=['ds_msclap_model', 'sv_msclap_model'],
        gating_method='feature_profile',
    ),
}


def create_gated_ensemble(
    models_dir: Path,
    config_name: str,
    verbose: bool = True,
) -> GatedEnsembleClassifier:
    """Create a gated ensemble from a pre-defined configuration.

    Args:
        models_dir: Directory containing expert model files
        config_name: Name of config from GATED_ENSEMBLE_CONFIGS
        verbose: Print progress

    Returns:
        Configured GatedEnsembleClassifier
    """
    if config_name not in GATED_ENSEMBLE_CONFIGS:
        available = list(GATED_ENSEMBLE_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")

    config = GATED_ENSEMBLE_CONFIGS[config_name]
    return GatedEnsembleClassifier(models_dir, config, verbose=verbose)


def list_gated_ensembles() -> List[str]:
    """List available gated ensemble configurations."""
    return list(GATED_ENSEMBLE_CONFIGS.keys())

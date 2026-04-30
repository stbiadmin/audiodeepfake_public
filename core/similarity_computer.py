"""Similarity computation module for pairwise embedding comparisons."""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import cdist, pdist, squareform


class SimilarityComputer:
    """Compute pairwise cosine similarities between embeddings.

    This class computes the N(N-1)/2 pairwise cosine similarities
    between all embedding pairs, which forms the basis for the
    statistical feature extraction.
    """

    def __init__(self, metric: str = 'cosine'):
        """Initialize the similarity computer.

        Args:
            metric: Distance metric to use ('cosine' or 'euclidean')
        """
        self.metric = metric

    def compute_pairwise(
        self,
        embeddings: np.ndarray,
        return_matrix: bool = False,
    ) -> np.ndarray:
        """Compute all pairwise similarities between embeddings.

        For N embeddings, computes N(N-1)/2 unique pairwise similarities
        (upper triangle of the similarity matrix, excluding diagonal).

        Args:
            embeddings: Array of shape (N, embedding_dim)
            return_matrix: If True, return full similarity matrix instead
                          of just the upper triangle values

        Returns:
            If return_matrix=False: 1D array of N(N-1)/2 similarity values
            If return_matrix=True: NxN similarity matrix
        """
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 embeddings to compute pairwise similarities")

        # Compute pairwise distances using scipy
        if self.metric == 'cosine':
            # pdist returns distances, convert to similarities
            distances = pdist(embeddings, metric='cosine')
            similarities = 1 - distances
        elif self.metric == 'euclidean':
            distances = pdist(embeddings, metric='euclidean')
            # Normalize euclidean distances to [0, 1] range
            max_dist = distances.max() if len(distances) > 0 else 1
            similarities = 1 - (distances / max_dist) if max_dist > 0 else distances
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        if return_matrix:
            return squareform(similarities)

        return similarities

    def compute_cross_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """Compute similarities between two sets of embeddings.

        Computes all pairs between embeddings1 and embeddings2
        (N1 x N2 similarities).

        Args:
            embeddings1: First set of embeddings (N1, embedding_dim)
            embeddings2: Second set of embeddings (N2, embedding_dim)

        Returns:
            2D array of shape (N1, N2) with pairwise similarities
        """
        if self.metric == 'cosine':
            distances = cdist(embeddings1, embeddings2, metric='cosine')
            return 1 - distances
        elif self.metric == 'euclidean':
            distances = cdist(embeddings1, embeddings2, metric='euclidean')
            max_dist = distances.max() if distances.size > 0 else 1
            return 1 - (distances / max_dist) if max_dist > 0 else distances
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def get_similarity_count(self, n_embeddings: int) -> int:
        """Get the number of pairwise similarities for N embeddings.

        Args:
            n_embeddings: Number of embeddings

        Returns:
            Number of unique pairwise similarities: N(N-1)/2
        """
        return n_embeddings * (n_embeddings - 1) // 2

    def compute_with_stats(
        self,
        embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Compute similarities and basic statistics.

        Args:
            embeddings: Array of shape (N, embedding_dim)

        Returns:
            Tuple of (similarities, stats_dict)
        """
        similarities = self.compute_pairwise(embeddings)

        stats = {
            'n_embeddings': len(embeddings),
            'n_pairs': len(similarities),
            'min': float(similarities.min()),
            'max': float(similarities.max()),
            'mean': float(similarities.mean()),
            'std': float(similarities.std()),
        }

        return similarities, stats

    def compute_intra_identity_similarity(
        self,
        embeddings_dict: dict,
    ) -> dict:
        """Compute intra-identity similarities for multiple identities.

        For each identity, compute pairwise similarities between
        all embeddings belonging to that identity.

        Args:
            embeddings_dict: Dict mapping identity -> embeddings array

        Returns:
            Dict mapping identity -> similarities array
        """
        results = {}

        for identity, embeddings in embeddings_dict.items():
            if len(embeddings) >= 2:
                results[identity] = self.compute_pairwise(embeddings)

        return results

    def compute_cross_identity_similarity(
        self,
        embeddings_dict: dict,
        sample_size: Optional[int] = None,
    ) -> np.ndarray:
        """Compute cross-identity similarities.

        Compute similarities between embeddings from different identities.

        Args:
            embeddings_dict: Dict mapping identity -> embeddings array
            sample_size: If provided, randomly sample this many pairs

        Returns:
            Array of cross-identity similarity values
        """
        identities = list(embeddings_dict.keys())
        all_cross_similarities = []

        for i, id1 in enumerate(identities):
            for id2 in identities[i + 1:]:
                cross_sim = self.compute_cross_similarity(
                    embeddings_dict[id1],
                    embeddings_dict[id2],
                )
                all_cross_similarities.extend(cross_sim.flatten())

        similarities = np.array(all_cross_similarities)

        if sample_size is not None and len(similarities) > sample_size:
            indices = np.random.choice(
                len(similarities),
                size=sample_size,
                replace=False,
            )
            similarities = similarities[indices]

        return similarities

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings (useful before cosine similarity).

        Args:
            embeddings: Array of shape (N, embedding_dim)

        Returns:
            L2-normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return embeddings / norms

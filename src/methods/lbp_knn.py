from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors

from config import DEFAULT_KNN_NEIGHBORS, DEFAULT_KNN_METRIC
from src.methods.base import BaseIrisMethod
from src.preprocessing.normalization import NormalizedIris


class LbpKnnMethod(BaseIrisMethod):
    """
    Method 2: Local Binary Patterns (LBP) Histogram Feature Extraction + K-Nearest Neighbors.
    Computes spatial block LBP descriptors and classifies via gallery nearest-neighbor matching.
    Uses Cosine metric with per-block L2 normalization for enhanced texture invariance.
    """

    def __init__(
        self,
        grid_rows: int = 8,
        grid_cols: int = 16,
        n_bins: int = 32,
        k_neighbors: int = DEFAULT_KNN_NEIGHBORS,
        metric: str = "cosine"
    ):
        super().__init__(
            name="lbp_knn",
            score_type=f"lbp_knn_{metric}_distance",
            higher_is_better=False
        )
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_bins = n_bins
        self.k_neighbors = k_neighbors
        self.metric = metric

        self.nn_model: Optional[NearestNeighbors] = None
        self.gallery_features: Optional[np.ndarray] = None
        self.gallery_labels: List[str] = []

    def extract_features(self, normalized: NormalizedIris) -> np.ndarray:
        """
        Extract spatial block LBP histograms from normalized dual-eye iris strip.
        Incorporates segmentation mask weighting when available.
        """
        img = normalized.normalized_image
        mask = normalized.normalized_mask

        center = img[1:-1, 1:-1]
        lbp = np.zeros(center.shape, dtype=np.uint8)

        # 8 circular neighbors
        neighbors = [
            (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
            (0, 1, 3), (1, 1, 4), (1, 0, 5),
            (1, -1, 6), (0, -1, 7)
        ]
        for dy, dx, bit in neighbors:
            shifted = img[1 + dy : img.shape[0] - 1 + dy, 1 + dx : img.shape[1] - 1 + dx]
            lbp |= ((shifted >= center).astype(np.uint8) << bit)

        mask_c = mask[1:-1, 1:-1] if mask is not None else None

        # Spatial block histogram partitioning
        h, w = lbp.shape
        bh, bw = max(1, h // self.grid_rows), max(1, w // self.grid_cols)
        histograms = []

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                block = lbp[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                if mask_c is not None:
                    b_mask = mask_c[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                    valid = block[b_mask > 0]
                else:
                    valid = block.flatten()

                if len(valid) > 0:
                    hist, _ = np.histogram(valid, bins=self.n_bins, range=(0, 256))
                    hist = hist.astype(np.float32)
                    hist /= (np.linalg.norm(hist) + 1e-6)
                else:
                    hist = np.zeros(self.n_bins, dtype=np.float32)
                histograms.append(hist)

        feature_vector = np.concatenate(histograms)
        # Global L2 unit-norm scaling
        feature_vector /= (np.linalg.norm(feature_vector) + 1e-7)
        return feature_vector

    def compute_pairwise_score(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        """
        Pairwise distance between two LBP feature vectors.
        """
        if self.metric == "cosine":
            sim = float(np.dot(feature_a, feature_b))
            return max(0.0, 1.0 - sim)
        # Euclidean distance
        return float(np.linalg.norm(feature_a - feature_b))

    def fit(self, features: List[np.ndarray], labels: List[str]) -> "LbpKnnMethod":
        self.gallery_features = np.vstack(features)
        self.gallery_labels = list(labels)
        self.enrolled_subjects = sorted(list(set(labels)))

        k = min(self.k_neighbors, len(labels))
        self.nn_model = NearestNeighbors(
            n_neighbors=k,
            metric=self.metric,
            algorithm="auto"
        )
        self.nn_model.fit(self.gallery_features)
        self.is_fitted = True
        return self

    def predict_ranked(self, probe_feature: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_fitted or self.gallery_features is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        probe_vec = probe_feature.reshape(1, -1)

        # Compute distances to all enrolled gallery items
        if self.metric == "cosine":
            sims = np.dot(self.gallery_features, probe_vec.T).ravel()
            dists = np.maximum(0.0, 1.0 - sims)
        else:
            dists = np.linalg.norm(self.gallery_features - probe_vec, axis=1)

        # Aggregate minimum distance per unique subject
        subject_min_dist: Dict[str, float] = {}
        for dist, label in zip(dists, self.gallery_labels):
            if label not in subject_min_dist or dist < subject_min_dist[label]:
                subject_min_dist[label] = float(dist)

        scored_candidates = []
        for subj, dist in subject_min_dist.items():
            # Normalized similarity mapping: 1 / (1 + distance)
            sim = round(float(1.0 / (1.0 + dist)), 4)
            scored_candidates.append({
                "subject_id": subj,
                "raw_score": round(dist, 4),
                "score_type": self.score_type,
                "higher_is_better": self.higher_is_better,
                "normalized_similarity": sim,
                "is_match": self.is_match(dist)
            })

        # Ascending sort (lower distance is better)
        scored_candidates.sort(key=lambda x: x["raw_score"])

        for rank_idx, cand in enumerate(scored_candidates[:top_k], start=1):
            cand["rank"] = rank_idx

        return scored_candidates[:top_k]

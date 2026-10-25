from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import DEFAULT_SVM_C, DEFAULT_RANDOM_SEED
from src.methods.base import BaseIrisMethod
from src.preprocessing.normalization import NormalizedIris


class GaborSvmMethod(BaseIrisMethod):
    """
    Method 3: Multi-Filter 2D Gabor Energy Descriptors + Support Vector Machine (SVM).
    Convolves normalized iris strip with multi-scale Gabor filter bank,
    extracts regional energy statistics, normalizes with StandardScaler,
    and classifies via Linear Support Vector Machine.
    """

    def __init__(
        self,
        orientations: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        wavelengths: Tuple[float, ...] = (6.0, 12.0),
        grid_rows: int = 2,
        grid_cols: int = 8,
        c_param: float = DEFAULT_SVM_C,
        random_state: int = DEFAULT_RANDOM_SEED
    ):
        super().__init__(
            name="gabor_svm",
            score_type="svm_decision_score",
            higher_is_better=True
        )
        self.orientations = orientations
        self.wavelengths = wavelengths
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.c_param = c_param
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.svm_model: Optional[LinearSVC] = None
        self.gallery_features: Optional[np.ndarray] = None
        self.gallery_labels: List[str] = []

    def extract_features(self, normalized: NormalizedIris) -> np.ndarray:
        """
        Extract fixed-length Gabor energy statistics from normalized dual-eye iris.
        """
        img = normalized.normalized_image.astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)

        feats = []
        for theta in self.orientations:
            for lambd in self.wavelengths:
                k_real = cv2.getGaborKernel((15, 15), 3.0, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                k_imag = cv2.getGaborKernel((15, 15), 3.0, theta, lambd, 0.5, np.pi / 2, ktype=cv2.CV_32F)

                f_r = cv2.filter2D(img, cv2.CV_32F, k_real)
                f_i = cv2.filter2D(img, cv2.CV_32F, k_imag)
                magnitude = np.sqrt(f_r ** 2 + f_i ** 2)

                h, w = magnitude.shape
                bh, bw = h // self.grid_rows, w // self.grid_cols
                for r in range(self.grid_rows):
                    for c in range(self.grid_cols):
                        block = magnitude[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                        feats.append(float(block.mean()))
                        feats.append(float(block.std()))

        vec = np.array(feats, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-7
        return vec / norm

    def compute_pairwise_score(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        """
        Pairwise similarity between two Gabor feature vectors (cosine similarity).
        Higher is better.
        """
        return float(np.dot(feature_a, feature_b))

    def fit(self, features: List[np.ndarray], labels: List[str]) -> "GaborSvmMethod":
        self.gallery_features = np.vstack(features)
        self.gallery_labels = list(labels)
        self.enrolled_subjects = sorted(list(set(labels)))

        scaled_X = self.scaler.fit_transform(self.gallery_features)
        self.svm_model = LinearSVC(
            C=self.c_param,
            random_state=self.random_state,
            max_iter=3000,
            dual="auto"
        )
        self.svm_model.fit(scaled_X, self.gallery_labels)
        self.is_fitted = True
        return self

    def predict_ranked(self, probe_feature: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_fitted or self.svm_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        probe_scaled = self.scaler.transform(probe_feature.reshape(1, -1))
        decision_scores = self.svm_model.decision_function(probe_scaled)

        # In binary classification, decision_function returns 1D array of shape (1,)
        # In multiclass, returns (1, n_classes)
        classes = self.svm_model.classes_

        if len(classes) == 2:
            s = float(decision_scores[0])
            scores = {classes[1]: s, classes[0]: -s}
        else:
            scores = {cls: float(s) for cls, s in zip(classes, decision_scores[0])}

        scored_candidates = []
        for subj, raw_s in scores.items():
            # Normalized similarity mapping via logistic sigmoid: 1 / (1 + exp(-score))
            sim = round(float(1.0 / (1.0 + np.exp(-max(-10.0, min(10.0, raw_s))))), 4)
            scored_candidates.append({
                "subject_id": subj,
                "raw_score": round(raw_s, 4),
                "score_type": self.score_type,
                "higher_is_better": self.higher_is_better,
                "normalized_similarity": sim,
                "is_match": self.is_match(raw_s)
            })

        # Descending sort (higher decision score is better)
        scored_candidates.sort(key=lambda x: x["raw_score"], reverse=True)

        for rank_idx, cand in enumerate(scored_candidates[:top_k], start=1):
            cand["rank"] = rank_idx

        return scored_candidates[:top_k]

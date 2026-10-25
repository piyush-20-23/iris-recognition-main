from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.methods.base import BaseIrisMethod
from src.preprocessing.normalization import NormalizedIris


@dataclass
class GaborIrisCodeTemplate:
    code: np.ndarray  # uint8 binary array (e.g. 128 x 32)
    mask: np.ndarray  # uint8 binary mask array (128 x 32)


class GaborHammingMethod(BaseIrisMethod):
    """
    Method 1: 2D Multi-Scale Gabor Filter Bank + Masked Hamming Distance Matching.
    Extracts phase-quantized binary iris code and performs rotation-compensated
    masked Hamming distance comparison.
    """

    def __init__(
        self,
        orientations: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        wavelengths: Tuple[float, ...] = (6.0, 12.0),
        max_shift: int = 2,
        grid_height: int = 8,
        grid_width: int = 32
    ):
        super().__init__(
            name="gabor_hamming",
            score_type="masked_hamming_distance",
            higher_is_better=False
        )
        self.orientations = orientations
        self.wavelengths = wavelengths
        self.max_shift = max_shift
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Gallery: subject_id -> List[GaborIrisCodeTemplate]
        self.gallery: Dict[str, List[GaborIrisCodeTemplate]] = {}

    def extract_features(self, normalized: NormalizedIris) -> GaborIrisCodeTemplate:
        """
        Extract multi-scale, multi-orientation binary iris code and validity mask.
        """
        img = normalized.normalized_image.astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)

        codes = []
        for theta in self.orientations:
            for lambd in self.wavelengths:
                k_real = cv2.getGaborKernel((15, 15), 3.0, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                k_imag = cv2.getGaborKernel((15, 15), 3.0, theta, lambd, 0.5, np.pi / 2, ktype=cv2.CV_32F)

                f_r = cv2.filter2D(img, cv2.CV_32F, k_real)
                f_i = cv2.filter2D(img, cv2.CV_32F, k_imag)

                b_r = cv2.resize(f_r, (self.grid_width, self.grid_height), interpolation=cv2.INTER_AREA)
                b_i = cv2.resize(f_i, (self.grid_width, self.grid_height), interpolation=cv2.INTER_AREA)

                codes.append((b_r >= 0).astype(np.uint8))
                codes.append((b_i >= 0).astype(np.uint8))

        code = np.concatenate(codes, axis=0)

        # Downsample normalized mask to matching grid
        raw_mask = (normalized.normalized_mask > 0).astype(np.float32)
        b_mask = (cv2.resize(raw_mask, (self.grid_width, self.grid_height), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
        mask = np.tile(b_mask, (len(codes), 1))

        return GaborIrisCodeTemplate(code=code, mask=mask)

    def compute_pairwise_score(
        self,
        feature_a: GaborIrisCodeTemplate,
        feature_b: GaborIrisCodeTemplate
    ) -> float:
        """
        Computes minimal masked Hamming distance with rotation compensation
        evaluated independently for left and right eyes.
        """
        code_a, mask_a = feature_a.code, feature_a.mask
        code_b, mask_b = feature_b.code, feature_b.mask

        half_w = self.grid_width // 2
        dists = []

        for s1 in range(-self.max_shift, self.max_shift + 1):
            for s2 in range(-self.max_shift, self.max_shift + 1):
                # Roll eye 1 and eye 2 independently
                c2_left = np.roll(code_b[:, :half_w], s1, axis=1)
                m2_left = np.roll(mask_b[:, :half_w], s1, axis=1)

                c2_right = np.roll(code_b[:, half_w:], s2, axis=1)
                m2_right = np.roll(mask_b[:, half_w:], s2, axis=1)

                c2_rolled = np.hstack([c2_left, c2_right])
                m2_rolled = np.hstack([m2_left, m2_right])

                valid = (mask_a & m2_rolled) > 0
                tot_valid = np.count_nonzero(valid)
                if tot_valid < 10:
                    dists.append(0.5)
                else:
                    diff = (code_a != c2_rolled) & valid
                    dists.append(np.count_nonzero(diff) / tot_valid)

        return float(min(dists)) if dists else 0.5

    def fit(self, features: List[GaborIrisCodeTemplate], labels: List[str]) -> "GaborHammingMethod":
        self.gallery.clear()
        self.enrolled_subjects = sorted(list(set(labels)))
        for feat, label in zip(features, labels):
            if label not in self.gallery:
                self.gallery[label] = []
            self.gallery[label].append(feat)
        self.is_fitted = True
        return self

    def predict_ranked(
        self,
        probe_feature: GaborIrisCodeTemplate,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        scored_candidates = []
        for subj, templates in self.gallery.items():
            # Best match among all enrolled templates for this subject
            min_dist = min(self.compute_pairwise_score(probe_feature, t) for t in templates)
            # Normalized similarity mapping: [0, 1]
            similarity = round(max(0.0, min(1.0, 1.0 - (min_dist / 0.5))), 4)
            scored_candidates.append({
                "subject_id": subj,
                "raw_score": round(min_dist, 4),
                "score_type": self.score_type,
                "higher_is_better": self.higher_is_better,
                "normalized_similarity": similarity,
                "is_match": self.is_match(min_dist)
            })

        # Sort ascending (lower Hamming distance is better)
        scored_candidates.sort(key=lambda x: x["raw_score"])

        for rank_idx, cand in enumerate(scored_candidates[:top_k], start=1):
            cand["rank"] = rank_idx

        return scored_candidates[:top_k]

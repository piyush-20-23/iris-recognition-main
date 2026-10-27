from typing import Any, Dict, List
import numpy as np

from src.methods.base import BaseIrisMethod
from src.methods.lbp_knn import LbpKnnMethod
from src.methods.gabor_svm import GaborSvmMethod
from src.preprocessing.normalization import NormalizedIris


class EnsembleFusionMethod(BaseIrisMethod):
    """
    Method 4: Multi-Modal Biometric Score-Level Ensemble Fusion.
    Fuses multi-scale Gabor filter energy representations and spatial block Local Binary Patterns (LBP).
    Significantly outperforms single-modality algorithms by combining phase/frequency responses
    with fine micro-texture characteristics.
    """

    def __init__(
        self,
        svm_weight: float = 0.55,
        lbp_weight: float = 0.45
    ):
        super().__init__(
            name="ensemble_fusion",
            score_type="ensemble_confidence_score",
            higher_is_better=True
        )
        self.svm_weight = svm_weight
        self.lbp_weight = lbp_weight

        self.method_svm = GaborSvmMethod()
        self.method_lbp = LbpKnnMethod()

    def extract_features(self, normalized: NormalizedIris) -> Dict[str, Any]:
        """
        Extracts both Gabor SVM and LBP feature vectors from the normalized strip.
        """
        feat_svm = self.method_svm.extract_features(normalized)
        feat_lbp = self.method_lbp.extract_features(normalized)
        return {
            "svm": feat_svm,
            "lbp": feat_lbp
        }

    def compute_pairwise_score(self, feature_a: Dict[str, Any], feature_b: Dict[str, Any]) -> float:
        """
        Combined similarity score between two multi-modal features.
        """
        s_svm = self.method_svm.compute_pairwise_score(feature_a["svm"], feature_b["svm"])
        d_lbp = self.method_lbp.compute_pairwise_score(feature_a["lbp"], feature_b["lbp"])
        s_lbp = 1.0 / (1.0 + d_lbp)
        return float(self.svm_weight * s_svm + self.lbp_weight * s_lbp)

    def fit(self, features: List[Dict[str, Any]], labels: List[str]) -> "EnsembleFusionMethod":
        feats_svm = [f["svm"] for f in features]
        feats_lbp = [f["lbp"] for f in features]

        self.method_svm.fit(feats_svm, labels)
        self.method_lbp.fit(feats_lbp, labels)

        self.enrolled_subjects = sorted(list(set(labels)))
        self.is_fitted = True
        return self

    def predict_ranked(self, probe_feature: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        svm_cands = self.method_svm.predict_ranked(probe_feature["svm"], top_k=len(self.enrolled_subjects))
        lbp_cands = self.method_lbp.predict_ranked(probe_feature["lbp"], top_k=len(self.enrolled_subjects))

        svm_map = {c["subject_id"]: c["normalized_similarity"] for c in svm_cands}
        lbp_map = {c["subject_id"]: c["normalized_similarity"] for c in lbp_cands}

        scored = []
        for subj in self.enrolled_subjects:
            s_svm = svm_map.get(subj, 0.0)
            s_lbp = lbp_map.get(subj, 0.0)
            combined_sim = round(float(self.svm_weight * s_svm + self.lbp_weight * s_lbp), 4)

            scored.append({
                "subject_id": subj,
                "raw_score": combined_sim,
                "score_type": self.score_type,
                "higher_is_better": self.higher_is_better,
                "normalized_similarity": combined_sim,
                "is_match": self.is_match(combined_sim)
            })

        # Sort descending by fused confidence
        scored.sort(key=lambda x: x["raw_score"], reverse=True)

        for rank_idx, cand in enumerate(scored[:top_k], start=1):
            cand["rank"] = rank_idx

        return scored[:top_k]

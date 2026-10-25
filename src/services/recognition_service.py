import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from config import DATASET_DIR, DEFAULT_ENROLLMENT_SIZES, DEFAULT_RANDOM_SEED
from src.data.inventory import DatasetInventory
from src.data.labels import IrisSampleMetadata
from src.evaluation.protocols import create_leakage_safe_split
from src.evaluation.verification import run_verification_evaluation
from src.methods.base import BaseIrisMethod
from src.methods.gabor_hamming import GaborHammingMethod
from src.methods.lbp_knn import LbpKnnMethod
from src.methods.gabor_svm import GaborSvmMethod
from src.preprocessing.image_io import load_image, load_mask, ImageReadError
from src.preprocessing.quality import assess_image_quality
from src.preprocessing.normalization import extract_and_normalize


class RecognitionService:
    """
    Central biometric recognition orchestration service.
    Completely decoupled from web/Flask frameworks.
    Maintains pre-enrolled galleries and provides rapid 1:N identification and 1:1 verification.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.inventory = DatasetInventory(self.dataset_dir)
        self.methods: Dict[str, BaseIrisMethod] = {
            "gabor_hamming": GaborHammingMethod(),
            "lbp_knn": LbpKnnMethod(),
            "gabor_svm": GaborSvmMethod()
        }
        self.is_enrolled: bool = False
        self.enrolled_subjects: List[str] = []
        self.enrollment_size: int = 3
        self.cached_summary: Optional[Dict[str, Any]] = None
        self.cached_evaluation_results: Optional[Dict[str, Any]] = None

    def initialize_and_enroll(
        self,
        enrollment_size: int = 3,
        max_subjects: Optional[int] = 30,
        seed: int = DEFAULT_RANDOM_SEED
    ) -> None:
        """
        Loads dataset inventory and enrolls the gallery into all 3 methods.
        Also derives validation operating thresholds for verification.
        """
        self.cached_summary = self.inventory.run_inventory(sample_decode_limit=50)
        split = create_leakage_safe_split(
            self.inventory.subject_samples,
            enrollment_size=enrollment_size,
            val_samples_per_subject=2,
            max_subjects=max_subjects,
            session_aware=True,
            seed=seed
        )

        self.enrollment_size = enrollment_size
        self.enrolled_subjects = sorted(list(split.enrollment_samples.keys()))

        # Enroll gallery templates for each method
        for method_name, method in self.methods.items():
            feats = []
            labels = []
            for subj, samples in split.enrollment_samples.items():
                for s in samples:
                    img = load_image(s.image_path)
                    mask = load_mask(s.mask_path) if s.mask_path else None
                    norm, _ = extract_and_normalize(img, mask)
                    feat = method.extract_features(norm)
                    feats.append(feat)
                    labels.append(subj)
            method.fit(feats, labels)

            # Load validation-derived threshold from benchmark summary if available
            eval_summary_file = Path(__file__).resolve().parent.parent.parent / "outputs" / "results" / "evaluation_summary.json"
            threshold_set = False
            if eval_summary_file.exists():
                try:
                    with open(eval_summary_file, "r", encoding="utf-8") as f:
                        summary_json = json.load(f)
                    for vr in summary_json.get("verification_results", []):
                        if vr.get("method_name") == method_name and "validation_derived_threshold" in vr:
                            method.set_validation_threshold(vr["validation_derived_threshold"])
                            threshold_set = True
                            break
                except Exception:
                    pass

            if not threshold_set:
                default_thresh = 0.97 if method.higher_is_better else (0.25 if method_name == "gabor_hamming" else 0.31)
                method.set_validation_threshold(default_thresh)

        self.is_enrolled = True

    def identify(
        self,
        image_source: Union[str, Path, bytes, np.ndarray],
        mask_source: Optional[Union[str, Path, bytes, np.ndarray]] = None,
        method_name: str = "all",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        1:N Identification for an input image.
        Accepts raw image array, filepath, or byte stream.
        """
        if not self.is_enrolled:
            self.initialize_and_enroll()

        img = load_image(image_source)
        mask = load_mask(mask_source) if mask_source is not None else None

        quality = assess_image_quality(img)
        norm, seg = extract_and_normalize(img, mask)

        available_methods = list(self.methods.keys())
        target_methods = available_methods if method_name == "all" else [method_name]

        if method_name != "all" and method_name not in self.methods:
            raise ValueError(f"Unknown method '{method_name}'. Supported: {available_methods} or 'all'")

        method_results: Dict[str, Any] = {}
        for m_name in target_methods:
            m = self.methods[m_name]
            feat = m.extract_features(norm)
            ranked_cands = m.predict_ranked(feat, top_k=top_k)
            method_results[m_name] = {
                "method_name": m.name,
                "score_type": m.score_type,
                "higher_is_better": m.higher_is_better,
                "validation_threshold": m.validation_threshold,
                "top_candidates": ranked_cands,
                "predicted_subject": ranked_cands[0]["subject_id"] if ranked_cands else None
            }

        return {
            "status": "success",
            "quality": {
                "is_usable": quality.is_usable,
                "sharpness": quality.sharpness,
                "contrast": quality.contrast,
                "mean_brightness": quality.mean_brightness,
                "issues": quality.issues
            },
            "segmentation": {
                "success": seg.success,
                "method": seg.method,
                "detected_eyes_count": len(seg.regions),
                "message": seg.message
            },
            "enrolled_gallery_size": len(self.enrolled_subjects),
            "results": method_results
        }

    def verify(
        self,
        image_source: Union[str, Path, bytes, np.ndarray],
        claimed_subject_id: str,
        mask_source: Optional[Union[str, Path, bytes, np.ndarray]] = None,
        method_name: str = "gabor_hamming"
    ) -> Dict[str, Any]:
        """
        1:1 Verification against a claimed identity.
        Uses validation-derived operating threshold.
        """
        if not self.is_enrolled:
            self.initialize_and_enroll()

        if method_name not in self.methods:
            raise ValueError(f"Unknown method '{method_name}'. Supported: {list(self.methods.keys())}")

        if claimed_subject_id not in self.enrolled_subjects:
            return {
                "status": "error",
                "message": f"Claimed subject ID '{claimed_subject_id}' is not in enrolled gallery."
            }

        img = load_image(image_source)
        mask = load_mask(mask_source) if mask_source is not None else None

        quality = assess_image_quality(img)
        norm, seg = extract_and_normalize(img, mask)

        m = self.methods[method_name]
        probe_feat = m.extract_features(norm)

        # Match against all enrolled subjects
        ranked = m.predict_ranked(probe_feat, top_k=len(self.enrolled_subjects))
        claimed_cand = next((c for c in ranked if c["subject_id"] == claimed_subject_id), None)

        if claimed_cand is None:
            raw_score = 1.0 if not m.higher_is_better else 0.0
            is_match = False
            sim = 0.0
            rank = len(self.enrolled_subjects)
        else:
            raw_score = claimed_cand["raw_score"]
            is_match = bool(claimed_cand["is_match"]) if claimed_cand["is_match"] is not None else False
            sim = claimed_cand["normalized_similarity"]
            rank = claimed_cand["rank"]

        return {
            "status": "success",
            "claimed_subject_id": claimed_subject_id,
            "method_name": m.name,
            "raw_score": raw_score,
            "score_type": m.score_type,
            "higher_is_better": m.higher_is_better,
            "normalized_similarity": sim,
            "rank": rank,
            "validation_threshold": m.validation_threshold,
            "is_verified_match": is_match,
            "decision": "MATCH" if is_match else "NO_MATCH",
            "quality": {
                "is_usable": quality.is_usable,
                "sharpness": quality.sharpness,
                "issues": quality.issues
            }
        }

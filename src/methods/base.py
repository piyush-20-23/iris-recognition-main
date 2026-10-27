from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from src.preprocessing.normalization import NormalizedIris


class BaseIrisMethod(ABC):
    """
    Abstract base class for all iris recognition algorithms.
    Guarantees consistent input/output contracts across Gabor+Hamming, LBP+KNN, and Gabor+SVM.
    """

    def __init__(self, name: str, score_type: str, higher_is_better: bool):
        self.name = name
        self.score_type = score_type
        self.higher_is_better = higher_is_better
        self.is_fitted: bool = False
        self.enrolled_subjects: List[str] = []
        self.validation_threshold: Optional[float] = None

    @abstractmethod
    def extract_features(self, normalized: NormalizedIris) -> Any:
        """Extract method-specific biometric representation from a normalized iris."""
        pass

    @abstractmethod
    def fit(self, features: List[Any], labels: List[str]) -> "BaseIrisMethod":
        """Enroll gallery templates or train classifier."""
        pass

    @abstractmethod
    def predict_ranked(self, probe_feature: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Match a probe feature against the gallery.
        Returns a list of ranked candidate dictionaries:
        [
            {
                "rank": int,
                "subject_id": str,
                "raw_score": float,
                "score_type": str,
                "higher_is_better": bool,
                "normalized_similarity": float,  # Always in [0, 1] where 1.0 is best
                "is_match": Optional[bool]       # Populated if validation_threshold is set
            },
            ...
        ]
        """
        pass

    @abstractmethod
    def compute_pairwise_score(self, feature_a: Any, feature_b: Any) -> float:
        """
        Computes 1:1 raw comparison score between two individual features.
        """
        pass

    def set_validation_threshold(self, threshold: float) -> None:
        """Sets validation-derived operating threshold for verification / open-set matching."""
        self.validation_threshold = float(threshold)

    def is_match(self, raw_score: float) -> Optional[bool]:
        """
        Determines whether a raw score passes the validation-derived threshold.
        """
        if self.validation_threshold is None:
            return None
        if self.higher_is_better:
            return raw_score >= self.validation_threshold
        else:
            return raw_score <= self.validation_threshold

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class VerificationMetrics:
    threshold: float
    far: float  # False Accept Rate: impostors accepted / total impostors
    frr: float  # False Reject Rate: genuines rejected / total genuines
    eer: float  # Equal Error Rate
    eer_threshold: float
    genuine_scores: List[float] = field(default_factory=list)
    impostor_scores: List[float] = field(default_factory=list)


def compute_far_frr(
    genuine_scores: List[float],
    impostor_scores: List[float],
    threshold: float,
    higher_is_better: bool = False
) -> Tuple[float, float]:
    """
    Computes FAR and FRR at a specified operating threshold.
    
    FAR = impostors accepted / total impostors
    FRR = genuines rejected / total genuines
    """
    g_arr = np.array(genuine_scores, dtype=np.float64)
    i_arr = np.array(impostor_scores, dtype=np.float64)

    if len(g_arr) == 0 or len(i_arr) == 0:
        return 0.0, 0.0

    if higher_is_better:
        # Accepted if score >= threshold
        far = float(np.mean(i_arr >= threshold))
        frr = float(np.mean(g_arr < threshold))
    else:
        # Distance metric: accepted if distance <= threshold
        far = float(np.mean(i_arr <= threshold))
        frr = float(np.mean(g_arr > threshold))

    return round(far, 6), round(frr, 6)


def compute_roc_and_eer(
    genuine_scores: List[float],
    impostor_scores: List[float],
    higher_is_better: bool = False,
    num_thresholds: int = 200
) -> Dict[str, Any]:
    """
    Computes FAR/FRR curve across threshold sweep and determines Equal Error Rate (EER).
    """

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    if len(all_scores) == 0:
        return {"eer": 0.0, "eer_threshold": 0.0, "thresholds": [], "far": [], "frr": []}

    min_s, max_s = float(np.min(all_scores)), float(np.max(all_scores))
    thresholds = np.linspace(min_s, max_s, num_thresholds)

    fars = []
    frrs = []

    for t in thresholds:
        far, frr = compute_far_frr(genuine_scores, impostor_scores, float(t), higher_is_better)
        fars.append(far)
        frrs.append(frr)

    # Find threshold where |FAR - FRR| is minimal (EER)
    diffs = np.abs(np.array(fars) - np.array(frrs))
    idx_eer = int(np.argmin(diffs))
    eer = float((fars[idx_eer] + frrs[idx_eer]) / 2.0)
    eer_threshold = float(thresholds[idx_eer])

    return {
        "eer": round(eer, 6),
        "eer_threshold": round(eer_threshold, 6),
        "thresholds": [round(float(t), 5) for t in thresholds],
        "far": [round(float(f), 5) for f in fars],
        "frr": [round(float(f), 5) for f in frrs]
    }


def compute_rank_k_accuracy(
    predictions: List[List[Dict[str, Any]]],
    ground_truth_labels: List[str],
    k: int = 1
) -> float:
    """
    Computes Rank-k identification accuracy.
    Rank-1: true subject is top-1 ranked candidate.
    """
    if not predictions:
        return 0.0

    correct = 0
    total = len(predictions)

    for ranked_candidates, true_label in zip(predictions, ground_truth_labels):
        top_k_subjects = [cand["subject_id"] for cand in ranked_candidates[:k]]
        if true_label in top_k_subjects:
            correct += 1

    return round(float(correct / total), 4)

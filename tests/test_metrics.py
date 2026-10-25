import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_far_frr,
    compute_roc_and_eer,
    compute_rank_k_accuracy
)


def test_hand_checkable_far_frr_distance():
    """
    Test FAR and FRR on deterministic, hand-checkable values where lower score is better (distance).
    """
    genuine = [0.1, 0.2, 0.3]
    impostor = [0.4, 0.5, 0.6]

    # At threshold 0.35:
    # Genuine attempts <= 0.35: all 3 accepted -> FRR = 0 / 3 = 0.0
    # Impostor attempts <= 0.35: 0 accepted -> FAR = 0 / 3 = 0.0
    far, frr = compute_far_frr(genuine, impostor, threshold=0.35, higher_is_better=False)
    assert far == 0.0
    assert frr == 0.0

    # At threshold 0.25:
    # Genuine <= 0.25: [0.1, 0.2] accepted, [0.3] rejected -> FRR = 1 / 3 = 0.333333
    # Impostor <= 0.25: none accepted -> FAR = 0 / 3 = 0.0
    far, frr = compute_far_frr(genuine, impostor, threshold=0.25, higher_is_better=False)
    assert far == 0.0
    assert pytest.approx(frr, 1e-4) == 0.333333

    # At threshold 0.45:
    # Impostor <= 0.45: [0.4] accepted -> FAR = 1 / 3 = 0.333333
    # Genuine <= 0.45: all accepted -> FRR = 0.0
    far, frr = compute_far_frr(genuine, impostor, threshold=0.45, higher_is_better=False)
    assert pytest.approx(far, 1e-4) == 0.333333
    assert frr == 0.0


def test_hand_checkable_far_frr_similarity():
    """
    Test FAR and FRR where higher score is better (similarity / probability).
    """
    genuine = [0.9, 0.8, 0.7]
    impostor = [0.3, 0.2, 0.1]

    # Threshold 0.75:
    # Genuine >= 0.75: [0.9, 0.8] accepted, [0.7] rejected -> FRR = 1/3
    # Impostor >= 0.75: none accepted -> FAR = 0.0
    far, frr = compute_far_frr(genuine, impostor, threshold=0.75, higher_is_better=True)
    assert far == 0.0
    assert pytest.approx(frr, 1e-4) == 0.333333


def test_rank_k_accuracy():
    predictions = [
        [{"subject_id": "C1", "rank": 1}, {"subject_id": "C2", "rank": 2}],
        [{"subject_id": "C3", "rank": 1}, {"subject_id": "C2", "rank": 2}],
        [{"subject_id": "C5", "rank": 1}, {"subject_id": "C4", "rank": 2}],
    ]
    ground_truth = ["C1", "C2", "C4"]

    # Rank-1: only probe 0 matches ("C1") -> 1/3 = 0.3333
    rank_1 = compute_rank_k_accuracy(predictions, ground_truth, k=1)
    assert pytest.approx(rank_1, 1e-3) == 0.3333

    # Rank-2: probe 0 matches "C1", probe 1 matches "C2" (rank 2), probe 2 matches "C4" (rank 2) -> 3/3 = 1.0
    rank_2 = compute_rank_k_accuracy(predictions, ground_truth, k=2)
    assert rank_2 == 1.0

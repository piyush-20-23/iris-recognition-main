import random
from typing import Any, Dict, List, Tuple

from src.evaluation.metrics import compute_far_frr, compute_roc_and_eer
from src.evaluation.protocols import DatasetSplit
from src.methods.base import BaseIrisMethod
from src.preprocessing.image_io import load_image, load_mask
from src.preprocessing.normalization import extract_and_normalize


def _extract_split_features(
    method: BaseIrisMethod,
    sample_dict: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """
    Extracts features for all samples in a subject dictionary.
    """
    features_by_subject: Dict[str, List[Any]] = {}
    for subj, samples in sample_dict.items():
        feats = []
        for s in samples:
            img = load_image(s.image_path)
            mask = load_mask(s.mask_path) if s.mask_path else None
            norm, _ = extract_and_normalize(img, mask)
            f = method.extract_features(norm)
            feats.append(f)
        if feats:
            features_by_subject[subj] = feats
    return features_by_subject


def _build_pairwise_scores(
    method: BaseIrisMethod,
    features_by_subject: Dict[str, List[Any]],
    max_impostors: int = 1000,
    seed: int = 42
) -> Tuple[List[float], List[float]]:
    """
    Generates genuine (same subject) and impostor (different subject) pairwise scores.
    """
    rng = random.Random(seed)
    genuine_scores = []
    impostor_scores = []

    # 1. Genuine pairs
    for subj, feats in features_by_subject.items():
        n = len(feats)
        for i in range(n):
            for j in range(i + 1, n):
                s = method.compute_pairwise_score(feats[i], feats[j])
                genuine_scores.append(s)

    # 2. Impostor pairs
    subjects = list(features_by_subject.keys())
    for _ in range(max_impostors):
        if len(subjects) < 2:
            break
        s1, s2 = rng.sample(subjects, 2)
        f1 = rng.choice(features_by_subject[s1])
        f2 = rng.choice(features_by_subject[s2])
        score = method.compute_pairwise_score(f1, f2)
        impostor_scores.append(score)

    return genuine_scores, impostor_scores


def run_verification_evaluation(
    method: BaseIrisMethod,
    split: DatasetSplit,
    max_impostors: int = 800
) -> Dict[str, Any]:
    """
    Evaluates 1:1 verification.
    Derives decision threshold strictly from validation data (preventing test leakage),
    then computes unbiased FAR and FRR on disjoint test partition.
    """
    # 1. Validation phase (threshold calibration)
    val_features = _extract_split_features(method, split.validation_samples)
    val_genuine, val_impostor = _build_pairwise_scores(
        method, val_features, max_impostors=max_impostors, seed=split.seed
    )
    val_roc = compute_roc_and_eer(val_genuine, val_impostor, method.higher_is_better)
    validation_threshold = val_roc["eer_threshold"]

    # Store derived threshold on model
    method.set_validation_threshold(validation_threshold)

    # 2. Test phase (unbiased evaluation)
    test_features = _extract_split_features(method, split.test_samples)
    test_genuine, test_impostor = _build_pairwise_scores(
        method, test_features, max_impostors=max_impostors, seed=split.seed + 1
    )

    test_far, test_frr = compute_far_frr(
        test_genuine, test_impostor, validation_threshold, method.higher_is_better
    )
    test_roc = compute_roc_and_eer(test_genuine, test_impostor, method.higher_is_better)

    return {
        "method_name": method.name,
        "validation_derived_threshold": validation_threshold,
        "test_far": test_far,
        "test_frr": test_frr,
        "test_eer": test_roc["eer"],
        "genuine_attempts": len(test_genuine),
        "impostor_attempts": len(test_impostor),
        "score_type": method.score_type,
        "higher_is_better": method.higher_is_better,
        "roc_curve": {
            "thresholds": test_roc["thresholds"],
            "far": test_roc["far"],
            "frr": test_roc["frr"]
        }
    }

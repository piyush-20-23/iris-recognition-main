import time
from typing import Any, Dict, List
import numpy as np

from src.evaluation.metrics import compute_rank_k_accuracy
from src.evaluation.protocols import DatasetSplit
from src.methods.base import BaseIrisMethod
from src.preprocessing.image_io import load_image, load_mask
from src.preprocessing.normalization import extract_and_normalize


def run_identification_evaluation(
    method: BaseIrisMethod,
    split: DatasetSplit,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluates 1:N closed-set identification on a disjoint test partition.
    """
    enrollment_features = []
    enrollment_labels = []
    extraction_latencies = []

    # 1. Feature extraction and fitting on enrollment set
    total_samples_evaluated = 0
    seg_successes = 0

    for subj, samples in split.enrollment_samples.items():
        for s in samples:
            total_samples_evaluated += 1
            t0 = time.perf_counter()
            img = load_image(s.image_path)
            mask = load_mask(s.mask_path) if s.mask_path else None
            norm, seg = extract_and_normalize(img, mask)
            if seg.success:
                seg_successes += 1
            feat = method.extract_features(norm)
            t1 = time.perf_counter()
            extraction_latencies.append((t1 - t0) * 1000.0)

            enrollment_features.append(feat)
            enrollment_labels.append(subj)

    fit_start = time.perf_counter()
    method.fit(enrollment_features, enrollment_labels)
    fit_duration_ms = (time.perf_counter() - fit_start) * 1000.0

    # 2. Evaluation on disjoint probe test set
    predictions: List[List[Dict[str, Any]]] = []
    ground_truth_labels: List[str] = []
    match_latencies = []

    for subj, samples in split.test_samples.items():
        for s in samples:
            total_samples_evaluated += 1
            img = load_image(s.image_path)
            mask = load_mask(s.mask_path) if s.mask_path else None
            norm, seg = extract_and_normalize(img, mask)
            if seg.success:
                seg_successes += 1
            probe_feat = method.extract_features(norm)

            m0 = time.perf_counter()
            ranked = method.predict_ranked(probe_feat, top_k=top_k)
            m1 = time.perf_counter()
            match_latencies.append((m1 - m0) * 1000.0)

            predictions.append(ranked)
            ground_truth_labels.append(subj)

    # 3. Compute accuracies
    rank_1_acc = compute_rank_k_accuracy(predictions, ground_truth_labels, k=1)
    rank_5_acc = compute_rank_k_accuracy(predictions, ground_truth_labels, k=min(5, top_k))
    usability_rate = round(float(seg_successes / max(1, total_samples_evaluated)), 4)

    return {
        "method_name": method.name,
        "enrollment_size": split.enrollment_size,
        "protocol": split.protocol_name,
        "enrolled_subjects_count": len(split.enrollment_samples),
        "total_probes_evaluated": len(ground_truth_labels),
        "rank_1_accuracy": rank_1_acc,
        "rank_5_accuracy": rank_5_acc,
        "input_usability_rate": usability_rate,
        "avg_extraction_latency_ms": round(float(np.mean(extraction_latencies)), 2) if extraction_latencies else 0.0,
        "avg_matching_latency_ms": round(float(np.mean(match_latencies)), 2) if match_latencies else 0.0,
        "fit_duration_ms": round(fit_duration_ms, 2)
    }

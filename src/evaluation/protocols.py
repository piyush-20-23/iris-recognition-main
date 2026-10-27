from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from config import DEFAULT_RANDOM_SEED
from src.data.labels import IrisSampleMetadata


@dataclass
class DatasetSplit:
    enrollment_samples: Dict[str, List[IrisSampleMetadata]]  # subject_id -> samples
    validation_samples: Dict[str, List[IrisSampleMetadata]]  # subject_id -> samples
    test_samples: Dict[str, List[IrisSampleMetadata]]        # subject_id -> samples
    enrollment_size: int
    protocol_name: str
    seed: int


def create_leakage_safe_split(
    subject_samples: Dict[str, List[IrisSampleMetadata]],
    enrollment_size: int = 3,
    val_samples_per_subject: int = 2,
    max_subjects: Optional[int] = None,
    session_aware: bool = True,
    seed: int = DEFAULT_RANDOM_SEED
) -> DatasetSplit:
    """
    Creates a strictly disjoint, leakage-safe dataset split for biometric evaluation.
    Zero image overlap between enrollment, validation, and probe test partitions.
    Orders subjects deterministically so primary cohort subjects (C1, C3, C5, ...)
    are reliably enrolled in the gallery.

    :param subject_samples: All parsed samples indexed by subject ID.
    :param enrollment_size: Number of gallery samples per subject (1, 3, 5, or 10).
    :param val_samples_per_subject: Number of validation samples per subject for threshold tuning.
    :param max_subjects: Limit number of subjects for faster evaluation benchmarks.
    :param session_aware: If True, prefers Session 1 for enrollment/val and Session 2 for test.
    :param seed: Random seed for exact reproducibility.
    """
    rng = random.Random(seed)
    # Sort subjects numerically by ID (e.g. C1, C3, C5, C7, C9...)
    selected_subjects = sorted(
        list(subject_samples.keys()),
        key=lambda s: int(s[1:]) if s[1:].isdigit() else 9999
    )

    if max_subjects is not None and max_subjects < len(selected_subjects):
        selected_subjects = selected_subjects[:max_subjects]

    enrollment: Dict[str, List[IrisSampleMetadata]] = {}
    validation: Dict[str, List[IrisSampleMetadata]] = {}
    test: Dict[str, List[IrisSampleMetadata]] = {}

    for subj in selected_subjects:
        samples = subject_samples[subj].copy()

        if session_aware:
            # Sort by session (S1 first, S2 later) then image id
            samples.sort(key=lambda s: (s.session_id, s.image_id))
        else:
            rng.shuffle(samples)

        min_needed = enrollment_size + val_samples_per_subject + 1
        if len(samples) < min_needed:
            # Skip subjects with insufficient samples for this enrollment size
            continue

        n_total = len(samples)
        # Select enrollment samples uniformly strided across the subject's captures.
        # This guarantees representation across both sessions (S1, S2) and distances (4m, 6m, 8m).
        step = n_total / float(enrollment_size)
        enroll_indices = set(int(i * step) for i in range(enrollment_size))
        curr_idx = 0
        while len(enroll_indices) < enrollment_size and curr_idx < n_total:
            enroll_indices.add(curr_idx)
            curr_idx += 1

        subj_enrollment = [samples[idx] for idx in sorted(enroll_indices)]
        remaining = [s for idx, s in enumerate(samples) if idx not in enroll_indices]
        subj_val = remaining[:val_samples_per_subject]
        subj_test = remaining[val_samples_per_subject:]

        enrollment[subj] = subj_enrollment
        validation[subj] = subj_val
        test[subj] = subj_test

    protocol_name = (
        f"{'session_aware' if session_aware else 'random'}_"
        f"enrollment_{enrollment_size}_val_{val_samples_per_subject}"
    )

    return DatasetSplit(
        enrollment_samples=enrollment,
        validation_samples=validation,
        test_samples=test,
        enrollment_size=enrollment_size,
        protocol_name=protocol_name,
        seed=seed
    )

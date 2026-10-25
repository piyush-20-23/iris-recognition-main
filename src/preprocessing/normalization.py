from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import (
    NORMALIZED_HEIGHT,
    NORMALIZED_WIDTH_PER_EYE,
    NORMALIZED_WIDTH_COMBINED,
    MASK_CLASS_IRIS
)
from src.segmentation.iris_region import IrisRegion, SegmentationResult


@dataclass
class NormalizedIris:
    normalized_image: np.ndarray  # Grayscale (H=64, W=512)
    normalized_mask: np.ndarray   # Binary uint8 0/1 (H=64, W=512)
    eye_count: int


def _normalize_single_region(
    region: IrisRegion,
    out_height: int = NORMALIZED_HEIGHT,
    out_width: int = NORMALIZED_WIDTH_PER_EYE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unrolls a single circular/annular iris patch into a standard rectangular strip
    using polar coordinate mapping (Daugman rubber-sheet transform).
    """
    patch = region.patch
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch.copy()

    h_p, w_p = gray.shape[:2]
    cx_p, cy_p = w_p / 2.0, h_p / 2.0
    max_radius = min(cx_p, cy_p)

    # Warp into polar coordinates: (width, height)
    # OpenCV warpPolar: dsize=(width, height) produces (height, width)
    polar_img = cv2.warpPolar(
        gray,
        (out_width, out_height),
        (cx_p, cy_p),
        max_radius,
        cv2.WARP_POLAR_LINEAR
    )

    # Apply CLAHE for illumination and contrast invariance in visible light
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    polar_img = clahe.apply(polar_img)

    # Warp mask if available
    if region.mask_patch is not None:
        mask_raw = (region.mask_patch == MASK_CLASS_IRIS).astype(np.uint8) * 255
        polar_mask = cv2.warpPolar(
            mask_raw,
            (out_width, out_height),
            (cx_p, cy_p),
            max_radius,
            cv2.INTER_NEAREST
        )
        polar_mask = (polar_mask > 127).astype(np.uint8)
    else:
        polar_mask = np.ones((out_height, out_width), dtype=np.uint8)

    return polar_img, polar_mask


def normalize_iris_regions(
    regions: List[IrisRegion],
    target_height: int = NORMALIZED_HEIGHT,
    target_width_per_eye: int = NORMALIZED_WIDTH_PER_EYE
) -> NormalizedIris:
    """
    Normalizes one or both extracted iris regions into a unified rectangular strip.
    If 2 regions are present: concatenates them horizontally into (64, 512).
    If 1 region is present: duplicates across eyes to preserve fixed descriptor dimensions.
    """
    if not regions:
        # Return fallback zero arrays
        zero_img = np.zeros((target_height, target_width_per_eye * 2), dtype=np.uint8)
        zero_mask = np.zeros((target_height, target_width_per_eye * 2), dtype=np.uint8)
        return NormalizedIris(zero_img, zero_mask, eye_count=0)

    strips = []
    masks = []

    for reg in regions[:2]:
        s_img, s_mask = _normalize_single_region(reg, target_height, target_width_per_eye)
        strips.append(s_img)
        masks.append(s_mask)

    if len(strips) == 1:
        # Duplicate single eye
        comb_img = np.hstack([strips[0], strips[0]])
        comb_mask = np.hstack([masks[0], masks[0]])
        eye_count = 1
    else:
        comb_img = np.hstack([strips[0], strips[1]])
        comb_mask = np.hstack([masks[0], masks[1]])
        eye_count = 2

    return NormalizedIris(
        normalized_image=comb_img,
        normalized_mask=comb_mask,
        eye_count=eye_count
    )


def extract_and_normalize(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[NormalizedIris, SegmentationResult]:
    """
    Convenience pipeline that runs segmentation followed by polar normalization.
    """
    from src.segmentation.iris_region import segment_iris
    seg_result = segment_iris(img, mask)
    norm = normalize_iris_regions(seg_result.regions)
    return norm, seg_result

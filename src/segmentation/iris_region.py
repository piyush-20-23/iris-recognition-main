from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import MASK_CLASS_IRIS


@dataclass
class IrisRegion:
    eye_idx: int
    center: Tuple[float, float]
    radius: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    patch: np.ndarray
    mask_patch: Optional[np.ndarray] = None


@dataclass
class SegmentationResult:
    success: bool
    regions: List[IrisRegion]
    method: str
    message: str


def _extract_crop_patch(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Safely crops a square region centered at (cx, cy) with side = 2 * r_crop.
    Applies boundary reflection/padding if near edges.
    """
    h_img, w_img = img.shape[:2]
    r_crop = int(max(radius * 1.35, 20))
    cx_i, cy_i = int(round(cx)), int(round(cy))

    x1 = cx_i - r_crop
    y1 = cy_i - r_crop
    x2 = cx_i + r_crop
    y2 = cy_i + r_crop

    # Compute padding if crop extends outside image boundaries
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h_img)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w_img)

    valid_y1 = max(0, y1)
    valid_y2 = min(h_img, y2)
    valid_x1 = max(0, x1)
    valid_x2 = min(w_img, x2)

    cropped = img[valid_y1:valid_y2, valid_x1:valid_x2]
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE
        )

    cropped_mask = None
    if mask is not None:
        m_cropped = mask[valid_y1:valid_y2, valid_x1:valid_x2]
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            m_cropped = cv2.copyMakeBorder(
                m_cropped, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        cropped_mask = m_cropped

    return cropped, cropped_mask


def segment_iris_from_mask(
    img: np.ndarray,
    mask: np.ndarray,
    min_area: int = 80
) -> SegmentationResult:
    """
    Extracts iris regions using the UBIPR ground-truth segmentation mask.
    """
    iris_bin = (mask == MASK_CLASS_IRIS).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(iris_bin)

    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            components.append({
                "stat": stats[i],
                "centroid": centroids[i],
                "area": area,
                "label_id": i
            })

    if not components:
        return SegmentationResult(
            success=False,
            regions=[],
            method="ground_truth_mask",
            message="No iris components with sufficient area found in mask."
        )

    # Sort components left to right (X coordinate)
    components.sort(key=lambda c: c["stat"][cv2.CC_STAT_LEFT])

    regions = []
    for eye_idx, comp in enumerate(components[:2]):  # At most 2 eyes
        cx, cy = comp["centroid"]
        st = comp["stat"]
        x, y, w, h = st[cv2.CC_STAT_LEFT], st[cv2.CC_STAT_TOP], st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT]
        radius = max(w, h) / 2.0

        patch, mask_patch = _extract_crop_patch(img, cx, cy, radius, mask)
        regions.append(IrisRegion(
            eye_idx=eye_idx,
            center=(float(cx), float(cy)),
            radius=float(radius),
            bbox=(int(x), int(y), int(w), int(h)),
            patch=patch,
            mask_patch=mask_patch
        ))

    return SegmentationResult(
        success=True,
        regions=regions,
        method="ground_truth_mask",
        message=f"Successfully extracted {len(regions)} iris region(s) using ground truth mask."
    )


def _detect_iris_in_subimage(
    sub_img: np.ndarray,
    x_offset: int = 0
) -> Optional[IrisRegion]:
    """
    Detects iris in an ocular subimage (e.g. left or right half of periocular face).
    """
    gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY) if sub_img.ndim == 3 else sub_img
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

    # In visible light UBIPR images, the pupil/iris is a dark central region
    # Adaptive thresholding or percentile thresholding
    p15 = np.percentile(blurred, 15)
    _, dark_thresh = cv2.threshold(blurred, p15, 255, cv2.THRESH_BINARY_INV)

    # Morphological open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter contours by size and aspect ratio
    h_sub, w_sub = gray.shape[:2]
    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > (w_sub * h_sub * 0.4):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < 12 or radius > min(w_sub, h_sub) * 0.45:
            continue
        # Circularity score
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        score = circularity * area
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        # Fallback: Hough Circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=25, minRadius=15, maxRadius=int(min(w_sub, h_sub) * 0.35)
        )
        if circles is not None and len(circles) > 0:
            c = circles[0][0]
            cx, cy, radius = c[0], c[1], c[2]
            patch, _ = _extract_crop_patch(sub_img, cx, cy, radius)
            return IrisRegion(
                eye_idx=0,
                center=(float(cx + x_offset), float(cy)),
                radius=float(radius),
                bbox=(int(cx - radius + x_offset), int(cy - radius), int(2 * radius), int(2 * radius)),
                patch=patch
            )
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    patch, _ = _extract_crop_patch(sub_img, cx, cy, radius)
    return IrisRegion(
        eye_idx=0,
        center=(float(cx + x_offset), float(cy)),
        radius=float(radius),
        bbox=(int(cx - radius + x_offset), int(cy - radius), int(2 * radius), int(2 * radius)),
        patch=patch
    )


def segment_iris_automated(img: np.ndarray) -> SegmentationResult:
    """
    Automated iris detector for unsegmented raw images.
    Handles both-eyes wide strips as well as single-eye crops.
    """
    h, w = img.shape[:2]
    aspect_ratio = w / h

    regions = []
    if aspect_ratio >= 1.6:
        # Wide periocular strip with both eyes
        mid_x = w // 2
        left_half = img[:, :mid_x]
        right_half = img[:, mid_x:]

        r1 = _detect_iris_in_subimage(left_half, x_offset=0)
        if r1:
            r1.eye_idx = 0
            regions.append(r1)

        r2 = _detect_iris_in_subimage(right_half, x_offset=mid_x)
        if r2:
            r2.eye_idx = 1
            regions.append(r2)
    else:
        # Single eye crop
        r = _detect_iris_in_subimage(img, x_offset=0)
        if r:
            r.eye_idx = 0
            regions.append(r)

    if not regions:
        # Fallback to center crops if automatic feature detection found no clean circle
        if aspect_ratio >= 1.6:
            r_est = min(h, w // 4) * 0.3
            patch1, _ = _extract_crop_patch(img, w * 0.28, h * 0.5, r_est)
            regions.append(IrisRegion(
                eye_idx=0, center=(w * 0.28, h * 0.5), radius=r_est,
                bbox=(int(w * 0.28 - r_est), int(h * 0.5 - r_est), int(2 * r_est), int(2 * r_est)),
                patch=patch1
            ))
            patch2, _ = _extract_crop_patch(img, w * 0.72, h * 0.5, r_est)
            regions.append(IrisRegion(
                eye_idx=1, center=(w * 0.72, h * 0.5), radius=r_est,
                bbox=(int(w * 0.72 - r_est), int(h * 0.5 - r_est), int(2 * r_est), int(2 * r_est)),
                patch=patch2
            ))
        else:
            r_est = min(h, w) * 0.35
            patch, _ = _extract_crop_patch(img, w * 0.5, h * 0.5, r_est)
            regions.append(IrisRegion(
                eye_idx=0, center=(w * 0.5, h * 0.5), radius=r_est,
                bbox=(int(w * 0.5 - r_est), int(h * 0.5 - r_est), int(2 * r_est), int(2 * r_est)),
                patch=patch
            ))

    return SegmentationResult(
        success=True,
        regions=regions,
        method="automated_detector",
        message=f"Automated segmentation located {len(regions)} eye region(s)."
    )


def segment_iris(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> SegmentationResult:
    """
    Main segmentation entry point. If mask is provided, uses ground truth segmentation;
    otherwise executes automated detector.
    """
    if mask is not None and np.any(mask == MASK_CLASS_IRIS):
        return segment_iris_from_mask(img, mask)
    return segment_iris_automated(img)

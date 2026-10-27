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
        # Fallback to automated detection if mask contains no valid iris component
        return segment_iris_automated(img)

    # Sort left-to-right (by x centroid)
    components.sort(key=lambda c: c["centroid"][0])

    regions = []
    for eye_idx, comp in enumerate(components[:2]):
        cx, cy = comp["centroid"]
        stat = comp["stat"]
        w_box, h_box = stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
        # Radius estimate based on equivalent circle area
        radius = float(np.sqrt(comp["area"] / np.pi))
        radius = max(radius, float(max(w_box, h_box) / 2.0))

        patch, mask_patch = _extract_crop_patch(img, cx, cy, radius, mask)
        regions.append(IrisRegion(
            eye_idx=eye_idx,
            center=(float(cx), float(cy)),
            radius=float(radius),
            bbox=(int(stat[cv2.CC_STAT_LEFT]), int(stat[cv2.CC_STAT_TOP]), int(w_box), int(h_box)),
            patch=patch,
            mask_patch=mask_patch
        ))

    return SegmentationResult(
        success=True,
        regions=regions,
        method="ground_truth_mask",
        message=f"Segmented {len(regions)} iris region(s) using ground-truth mask."
    )


def _detect_iris_in_subimage(
    sub_img: np.ndarray,
    x_offset: int = 0
) -> Optional[IrisRegion]:
    """
    Robust iris detector for an ocular subimage (e.g. left or right half of periocular face).
    Focuses on the anatomical eye band to prevent false detections in eyebrows and hair.
    """
    h_sub, w_sub = sub_img.shape[:2]

    # Focus on the anatomical eye band: excludes eyebrows (top 42%) and lower face (bottom 30%)
    # In UBIPr, iris center is strictly within Y in [0.45 * H, 0.65 * H], mean = 0.548 * H
    y1 = int(h_sub * 0.40)
    y2 = int(h_sub * 0.72)
    x1 = int(w_sub * 0.20)
    x2 = int(w_sub * 0.85)

    roi = sub_img[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blurred = cv2.GaussianBlur(gray_roi, (7, 7), 1.5)

    # In UBIPr, iris radius is strictly within [0.075 * H, 0.145 * H], mean = 0.111 * H
    min_r = max(14, int(h_sub * 0.075))
    max_r = max(min_r + 4, int(h_sub * 0.145))

    # Stage 1: Hough Circle Transform with inner pupil brightness verification
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min_r * 2,
        param1=45, param2=18, minRadius=min_r, maxRadius=max_r
    )

    best_c = None
    if circles is not None and len(circles) > 0:
        min_brightness = 999.0
        for c in circles[0]:
            cx_r, cy_r, r = c[0], c[1], c[2]
            # Verify pupil darkness inside inner circle
            mask_disk = np.zeros_like(gray_roi, dtype=np.uint8)
            cv2.circle(mask_disk, (int(cx_r), int(cy_r)), int(max(4, r * 0.4)), 255, -1)
            mean_b = cv2.mean(gray_roi, mask=mask_disk)[0]
            if mean_b < min_brightness:
                min_brightness = mean_b
                best_c = (cx_r, cy_r, r)

    # Stage 2: Dark circular contour search if Hough missed
    if best_c is None:
        p12 = np.percentile(blurred, 12)
        _, dark_thresh = cv2.threshold(blurred, p12, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * (min_r * 0.4) ** 2:
                continue
            (cx_r, cy_r), r = cv2.minEnclosingCircle(cnt)
            if r < min_r or r > max_r:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circ = 4 * np.pi * (area / (perimeter * perimeter))
            score = circ * area
            if score > best_score:
                best_score = score
                best_c = (cx_r, cy_r, r)

    # Stage 3: Anatomical center fallback
    if best_c is not None:
        cx = best_c[0] + x1
        cy = best_c[1] + y1
        radius = float(best_c[2])
    else:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = float((min_r + max_r) / 2.0)

    patch, _ = _extract_crop_patch(sub_img, cx, cy, radius)

    # Synthesize clean annular iris mask (Class 85: Iris ring; 0: Pupil & Sclera)
    mask_patch = np.zeros(patch.shape[:2], dtype=np.uint8)
    pcx, pcy = patch.shape[1] / 2.0, patch.shape[0] / 2.0
    cv2.circle(mask_patch, (int(pcx), int(pcy)), int(radius), MASK_CLASS_IRIS, -1)
    cv2.circle(mask_patch, (int(pcx), int(pcy)), int(radius * 0.32), 0, -1)  # Exclude pupil

    return IrisRegion(
        eye_idx=0,
        center=(float(cx + x_offset), float(cy)),
        radius=float(radius),
        bbox=(int(cx - radius + x_offset), int(cy - radius), int(2 * radius), int(2 * radius)),
        patch=patch,
        mask_patch=mask_patch
    )


def segment_iris_automated(img: np.ndarray) -> SegmentationResult:
    """
    Automated iris detector for unsegmented raw images.
    Handles both-eyes wide strips as well as single-eye crops.
    """
    h, w = img.shape[:2]
    aspect_ratio = w / h

    regions = []
    if aspect_ratio >= 1.4:
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
        # Fallback to center crops
        r_est = float(min(h, w // 4) * 0.28)
        patch1, _ = _extract_crop_patch(img, w * 0.28, h * 0.52, r_est)
        m1 = np.full(patch1.shape[:2], MASK_CLASS_IRIS, dtype=np.uint8)
        regions.append(IrisRegion(
            eye_idx=0, center=(w * 0.28, h * 0.52), radius=r_est,
            bbox=(int(w * 0.28 - r_est), int(h * 0.52 - r_est), int(2 * r_est), int(2 * r_est)),
            patch=patch1, mask_patch=m1
        ))
        patch2, _ = _extract_crop_patch(img, w * 0.72, h * 0.52, r_est)
        m2 = np.full(patch2.shape[:2], MASK_CLASS_IRIS, dtype=np.uint8)
        regions.append(IrisRegion(
            eye_idx=1, center=(w * 0.72, h * 0.52), radius=r_est,
            bbox=(int(w * 0.72 - r_est), int(h * 0.52 - r_est), int(2 * r_est), int(2 * r_est)),
            patch=patch2, mask_patch=m2
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

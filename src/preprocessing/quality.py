from dataclasses import dataclass, field
from typing import List
import cv2
import numpy as np


@dataclass
class ImageQualityReport:
    is_usable: bool
    sharpness: float
    contrast: float
    mean_brightness: float
    under_exposed_ratio: float
    over_exposed_ratio: float
    issues: List[str] = field(default_factory=list)


def assess_image_quality(
    img: np.ndarray,
    min_sharpness: float = 8.0,
    min_contrast: float = 10.0,
    min_brightness: float = 15.0,
    max_brightness: float = 240.0
) -> ImageQualityReport:
    """
    Assesses image usability and biometric quality metrics.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    issues = []
    # Sharpness via Laplacian variance
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < min_sharpness:
        issues.append(f"Low sharpness/blur (variance {sharpness:.1f} < {min_sharpness})")

    # Contrast via standard deviation
    contrast = float(gray.std())
    if contrast < min_contrast:
        issues.append(f"Low contrast (std {contrast:.1f} < {min_contrast})")

    # Brightness mean
    mean_brightness = float(gray.mean())
    if mean_brightness < min_brightness:
        issues.append(f"Image too dark (mean {mean_brightness:.1f} < {min_brightness})")
    elif mean_brightness > max_brightness:
        issues.append(f"Image overexposed (mean {mean_brightness:.1f} > {max_brightness})")

    # Saturation fractions
    under_ratio = float(np.mean(gray <= 5))
    over_ratio = float(np.mean(gray >= 250))

    if under_ratio > 0.40:
        issues.append("Excessive dark clipping")
    if over_ratio > 0.40:
        issues.append("Excessive highlight clipping")

    is_usable = (len(issues) == 0)

    return ImageQualityReport(
        is_usable=is_usable,
        sharpness=round(sharpness, 2),
        contrast=round(contrast, 2),
        mean_brightness=round(mean_brightness, 2),
        under_exposed_ratio=round(under_ratio, 4),
        over_exposed_ratio=round(over_ratio, 4),
        issues=issues
    )

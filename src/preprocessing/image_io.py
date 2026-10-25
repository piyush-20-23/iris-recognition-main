from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np


class ImageReadError(Exception):
    """Raised when an image cannot be read, decoded, or validated."""
    pass


def load_image(
    source: Union[str, Path, bytes, np.ndarray],
    as_gray: bool = False
) -> np.ndarray:
    """
    Safely load an image from a filepath, raw bytes, or existing numpy array.
    
    :param source: Path, byte string, or numpy ndarray.
    :param as_gray: If True, returns single-channel grayscale image.
    :return: uint8 numpy ndarray (H, W, C) or (H, W).
    :raises ImageReadError: If decoding fails or image is empty/corrupt.
    """
    if source is None:
        raise ImageReadError("Image source is None.")

    img: Optional[np.ndarray] = None

    if isinstance(source, np.ndarray):
        img = source.copy()
    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ImageReadError("Image byte stream is empty.")
        np_arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageReadError("Failed to decode image from byte buffer.")
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise ImageReadError(f"File not found: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageReadError(f"Failed to read image file: {p}")
    else:
        raise ImageReadError(f"Unsupported image source type: {type(source)}")

    if img is None or img.size == 0:
        raise ImageReadError("Decoded image has zero size.")

    if img.ndim < 2 or img.shape[0] < 20 or img.shape[1] < 20:
        raise ImageReadError(f"Image dimensions too small: {img.shape}")

    if as_gray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


def load_mask(
    source: Union[str, Path, bytes, np.ndarray]
) -> np.ndarray:
    """
    Safely load an 8-bit single channel segmentation mask.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 3:
            return source[:, :, 0].copy()
        return source.copy()
    elif isinstance(source, bytes):
        np_arr = np.frombuffer(source, np.uint8)
        mask = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ImageReadError("Failed to decode mask from bytes.")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise ImageReadError(f"Mask file not found: {p}")
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ImageReadError(f"Failed to read mask file: {p}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask
    else:
        raise ImageReadError(f"Unsupported mask source type: {type(source)}")

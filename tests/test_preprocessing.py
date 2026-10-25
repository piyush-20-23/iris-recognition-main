import numpy as np
import pytest
from src.preprocessing.image_io import load_image, load_mask, ImageReadError
from src.preprocessing.quality import assess_image_quality
from src.preprocessing.normalization import normalize_iris_regions, extract_and_normalize
from src.segmentation.iris_region import IrisRegion, segment_iris_automated


def test_load_image_valid_array():
    arr = np.ones((100, 200, 3), dtype=np.uint8) * 128
    img = load_image(arr)
    assert img.shape == (100, 200, 3)

    img_gray = load_image(arr, as_gray=True)
    assert img_gray.shape == (100, 200)


def test_load_image_corrupt_bytes():
    with pytest.raises(ImageReadError):
        load_image(b"not_an_image_bytes")


def test_load_image_empty():
    with pytest.raises(ImageReadError):
        load_image(b"")


def test_quality_assessment():
    # Sharp image
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    report = assess_image_quality(img)
    assert report.sharpness > 0
    assert report.mean_brightness > 0

    # Totally black image (too dark)
    dark_img = np.zeros((100, 100), dtype=np.uint8)
    dark_report = assess_image_quality(dark_img)
    assert not dark_report.is_usable
    assert any("dark" in iss.lower() for iss in dark_report.issues)


def test_normalize_iris_regions():
    dummy_patch1 = np.ones((80, 80, 3), dtype=np.uint8) * 100
    dummy_patch2 = np.ones((80, 80, 3), dtype=np.uint8) * 150
    reg1 = IrisRegion(0, (40.0, 40.0), 30.0, (10, 10, 60, 60), dummy_patch1)
    reg2 = IrisRegion(1, (40.0, 40.0), 30.0, (10, 10, 60, 60), dummy_patch2)

    norm = normalize_iris_regions([reg1, reg2])
    assert norm.normalized_image.shape == (64, 512)
    assert norm.normalized_mask.shape == (64, 512)
    assert norm.eye_count == 2

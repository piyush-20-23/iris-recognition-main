import pytest
from pathlib import Path
from src.data.labels import parse_sample_stem, parse_sample_metadata


def test_parse_valid_ubipr_filenames():
    parsed = parse_sample_stem("C101_S1_I1")
    assert parsed is not None
    assert parsed["subject_id"] == "C101"
    assert parsed["session_id"] == 1
    assert parsed["image_id"] == 1

    parsed2 = parse_sample_stem("C1_S2_I15")
    assert parsed2 is not None
    assert parsed2["subject_id"] == "C1"
    assert parsed2["session_id"] == 2
    assert parsed2["image_id"] == 15


def test_parse_invalid_filenames():
    assert parse_sample_stem("random_image") is None
    assert parse_sample_stem(".DS_Store") is None
    assert parse_sample_stem("C101_invalid") is None
    assert parse_sample_stem("._C101_S1_I1") is None


def test_parse_sample_metadata():
    p = Path("C101_S1_I10.jpg")
    meta = parse_sample_metadata(p)
    assert meta is not None
    assert meta.stem == "C101_S1_I10"
    assert meta.subject_id == "C101"
    assert meta.session_id == 1
    assert meta.image_id == 10

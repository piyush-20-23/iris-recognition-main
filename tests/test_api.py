import io
import cv2
import numpy as np
import pytest
import config
from app import app, recognition_service
from src.preprocessing.normalization import NormalizedIris


def _enroll_synthetic_gallery(service):
    dummy_img = np.random.randint(40, 200, (64, 512), dtype=np.uint8)
    dummy_mask = np.ones((64, 512), dtype=np.uint8)
    dummy_norm = NormalizedIris(dummy_img, dummy_mask, eye_count=2)

    subjects = ["C1", "C2", "C3", "C4", "C5"]
    for method_name, method in service.methods.items():
        feats = []
        labels = []
        for subj in subjects:
            for _ in range(2):
                feat = method.extract_features(dummy_norm)
                feats.append(feat)
                labels.append(subj)
        method.fit(feats, labels)
        default_thresh = 0.70 if method_name == "ensemble_fusion" else (0.97 if method.higher_is_better else 0.31)
        method.set_validation_threshold(default_thresh)
    service.enrolled_subjects = subjects
    service.is_enrolled = True


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    if config.DATASET_DIR.exists() and any(config.DATASET_DIR.glob("*.jpg")):
        recognition_service.initialize_and_enroll(enrollment_size=2, max_subjects=5)
    else:
        _enroll_synthetic_gallery(recognition_service)
    with app.test_client() as c:
        yield c


def _create_synthetic_image_bytes():
    img = np.random.randint(50, 200, (200, 500, 3), dtype=np.uint8)
    # Draw two dark pupil/iris circles
    cv2.circle(img, (150, 100), 30, (20, 20, 20), -1)
    cv2.circle(img, (350, 100), 30, (20, 20, 20), -1)
    success, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["is_gallery_enrolled"] is True


def test_dataset_summary_endpoint(client):
    res = client.get("/api/v1/dataset/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "summary" in data


def test_identify_missing_file(client):
    res = client.post("/api/v1/identify", data={})
    assert res.status_code == 400
    data = res.get_json()
    assert data["status"] == "error"
    assert "image" in data["message"].lower()


def test_identify_corrupt_file(client):
    data = {
        "image": (io.BytesIO(b"garbage_not_image"), "test.jpg"),
        "method": "all"
    }
    res = client.post("/api/v1/identify", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    res_data = res.get_json()
    assert res_data["status"] == "error"


def test_identify_invalid_method(client):
    img_bytes = _create_synthetic_image_bytes()
    data = {
        "image": (io.BytesIO(img_bytes), "test.jpg"),
        "method": "invalid_unknown_method"
    }
    res = client.post("/api/v1/identify", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    res_data = res.get_json()
    assert res_data["status"] == "error"


def test_identify_success(client):
    img_bytes = _create_synthetic_image_bytes()
    data = {
        "image": (io.BytesIO(img_bytes), "test.jpg"),
        "method": "all",
        "top_k": 3
    }
    res = client.post("/api/v1/identify", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["status"] == "success"
    assert "results" in res_data
    assert "gabor_hamming" in res_data["results"]
    assert "lbp_knn" in res_data["results"]
    assert "gabor_svm" in res_data["results"]
    assert "ensemble_fusion" in res_data["results"]


def test_verify_endpoint(client):
    img_bytes = _create_synthetic_image_bytes()
    claimed = recognition_service.enrolled_subjects[0]

    data = {
        "image": (io.BytesIO(img_bytes), "test.jpg"),
        "claimed_subject_id": claimed,
        "method": "ensemble_fusion"
    }
    res = client.post("/api/v1/verify", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["status"] == "success"
    assert res_data["claimed_subject_id"] == claimed
    assert "is_verified_match" in res_data
    assert res_data["decision"] in ["MATCH", "NO_MATCH"]

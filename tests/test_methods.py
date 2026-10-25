import numpy as np
import pytest
from src.methods.gabor_hamming import GaborHammingMethod
from src.methods.lbp_knn import LbpKnnMethod
from src.methods.gabor_svm import GaborSvmMethod
from src.preprocessing.normalization import NormalizedIris


@pytest.fixture
def dummy_normalized_iris():
    rng = np.random.RandomState(42)
    img = rng.randint(40, 200, (64, 512), dtype=np.uint8)
    mask = np.ones((64, 512), dtype=np.uint8)
    return NormalizedIris(img, mask, eye_count=2)


def test_gabor_hamming_lifecycle(dummy_normalized_iris):
    method = GaborHammingMethod()
    feat1 = method.extract_features(dummy_normalized_iris)
    feat2 = method.extract_features(dummy_normalized_iris)

    score_self = method.compute_pairwise_score(feat1, feat2)
    # Identical code should have Hamming distance == 0
    assert score_self == 0.0

    # Fit gallery
    method.fit([feat1, feat2], ["C1", "C2"])
    assert method.is_fitted

    ranked = method.predict_ranked(feat1, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["subject_id"] == "C1"
    assert ranked[0]["raw_score"] == 0.0
    assert ranked[0]["rank"] == 1
    assert ranked[0]["higher_is_better"] is False


def test_lbp_knn_lifecycle(dummy_normalized_iris):
    method = LbpKnnMethod()
    feat1 = method.extract_features(dummy_normalized_iris)
    feat2 = feat1 + np.random.randn(*feat1.shape) * 0.05
    feat2 /= np.linalg.norm(feat2)

    dist = method.compute_pairwise_score(feat1, feat1)
    assert pytest.approx(dist, 1e-4) == 0.0

    method.fit([feat1, feat2], ["C1", "C2"])
    assert method.is_fitted

    ranked = method.predict_ranked(feat1, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["subject_id"] == "C1"
    assert ranked[0]["rank"] == 1


def test_gabor_svm_lifecycle(dummy_normalized_iris):
    method = GaborSvmMethod()
    f1 = method.extract_features(dummy_normalized_iris)
    f2 = f1 + np.random.randn(*f1.shape) * 0.1
    f2 /= np.linalg.norm(f2)
    f3 = f1 + np.random.randn(*f1.shape) * 0.2
    f3 /= np.linalg.norm(f3)

    # Train SVM with multiple samples per class
    features = [f1, f1, f2, f2, f3, f3]
    labels = ["C1", "C1", "C2", "C2", "C3", "C3"]
    method.fit(features, labels)
    assert method.is_fitted

    ranked = method.predict_ranked(f1, top_k=3)
    assert len(ranked) == 3
    assert ranked[0]["rank"] == 1
    assert ranked[0]["higher_is_better"] is True
    # Highest decision score should be first
    assert ranked[0]["raw_score"] >= ranked[1]["raw_score"]

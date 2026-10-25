# UBIPr Dual-Eye Iris Recognition Biometric Research Prototype

A modular, research-grade iris biometric recognition system implemented with Python, OpenCV, scikit-learn, and Flask, benchmarked on the official **UBIPr (University of Beira Interior Periocular) Segmented Both-Eyes Dataset** (`ubipr2`).

The system implements three distinct biometric recognition algorithms, strict leakage-safe cross-session evaluation protocols, a versioned REST API, and an interactive browser frontend. The core recognition pipeline is decoupled from Flask, enabling direct future integration with Spring Boot backends and Android mobile applications.

> **Detailed Technical Guide:** For an exhaustive mathematical breakdown of the end-to-end pipeline and all three algorithms, refer to [**`WORKFLOW.md`**](WORKFLOW.md).

---

## 1. Architecture Overview

```
                      +---------------------------------------+
                      |       Raw Probe Input (Image/Mask)    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Image Quality Assessment (Laplacian)|
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Segmentation (GT Mask / Auto Detect)|
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Daugman Polar Normalization (64 x 512)|
                      +---------------------------------------+
                                          |
       +----------------------------------+----------------------------------+
       |                                  |                                  |
       v                                  v                                  v
+-------------------------+    +-----------------------+    +------------------------+
| Method 1: Gabor+Hamming |    | Method 2: LBP + KNN   |    | Method 3: Gabor + SVM  |
| 2D Multi-Scale Gabor    |    | Spatial Block LBP     |    | Multi-Filter Energy    |
| Masked Hamming Distance |    | L2 Normalized Descr.  |    | StandardScaler Scaling |
| Rotation Compensation   |    | Nearest Neighbors     |    | Linear SVM Multi-Class |
+-------------------------+    +-----------------------+    +------------------------+
       |                                  |                                  |
       +----------------------------------+----------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Decoupled Recognition Service       |
                      |   1:N Identification / 1:1 Verify     |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             +--------------------+               +-------------------+
             | Flask REST API v1  |               | Browser Web UI    |
             | (Android / Postman)|               | (Live Biometrics) |
             +--------------------+               +-------------------+
```

---

## 2. Dataset Audit & Ground Truth Semantics

The prototype was audited against the local dataset located at `c:\Users\Piyush\OneDrive\Desktop\ubipr2\both_eyes`:

- **Total Files Inventoried:** 11,020 items.
- **Operating Images:** 5,509 `.jpg` images (3-channel BGR, variable periocular resolutions).
- **Segmentation Masks:** 5,509 matching `.png` masks (100% paired, zero missing masks).
- **Ignored OS Artifacts:** `.DS_Store` and `._.DS_Store` (filtered safely; no source files modified).
- **Subjects:** 261 unique subjects (`C1` through `C...`).
- **Sessions:** Session 1 (3,914 images, 71.05%) and Session 2 (1,595 images, 28.95%).
- **Subject Sample Statistics:** Minimum 14 images per subject, maximum 33, mean 21.11. Because every subject has at least 14 images, enrollment sizes **1, 3, 5, and 10** are fully supported across all subjects.

### Mask Class Definitions
The 8-bit single-channel `.png` ground-truth masks delineate 4 anatomical classes:
- `0`: Background & Facial Skin
- `85`: **Iris Region** (Both Left and Right Eyes)
- `170`: **Sclera Region** (Medial and Lateral sides flanking the iris)
- `255`: **Eyebrows**

Visual audit report and 16-sample contact sheet with iris overlays:
- Markdown report: [`reports/dataset_audit.md`](reports/dataset_audit.md)
- JSON audit summary: [`reports/dataset_audit.json`](reports/dataset_audit.json)
- Contact sheet: [`reports/contact_sheet.jpg`](reports/contact_sheet.jpg)

---

## 3. Biometric Methods Implemented

### Method 1: Gabor Iris Code + Masked Hamming Distance (`gabor_hamming`)
- **Feature Extraction:** 2D Gabor wavelet filter bank (4 orientations: $0, \pi/4, \pi/2, 3\pi/4$; 2 wavelengths: $6.0, 12.0$). Real and imaginary responses are block-averaged ($8 \times 32$) and phase-quantized into a 1024-bit binary code.
- **Matching:** Masked Hamming distance discounting occluded/eyelid pixels.
- **Dual-Eye Rotation Compensation:** Independent cyclic shifting along horizontal columns for left and right eyes ($s \in [-2, 2]$) to handle head tilt and camera rotation.
- **Score:** Raw score is normalized Hamming distance $\in [0, 1]$ (lower is better). Normalized similarity is $1.0 - \text{HD}$.

### Method 2: LBP Spatial Block Histograms + K-Nearest Neighbors (`lbp_knn`)
- **Feature Extraction:** 8-neighbor circular Local Binary Patterns computed across normalized strips. Partitioned into a $4 \times 16$ spatial grid (64 blocks), each producing a 32-bin normalized histogram, yielding a 2048-dimensional unit-norm descriptor.
- **Matching:** Nearest Neighbor matching with Euclidean or Cosine distance.
- **Score:** Distance $\in [0, \infty)$ (lower is better). Normalized similarity is $\frac{1}{1 + \text{dist}}$.

### Method 3: Multi-Filter Gabor Energy Descriptors + Linear SVM (`gabor_svm`)
- **Feature Extraction:** Multi-scale Gabor filter bank convolved with normalized iris. Regional energy statistics (mean and standard deviation) extracted across a spatial block grid to form a 256-dimensional dense feature vector.
- **Scaling & Classification:** `StandardScaler` reproducible zero-mean unit-variance transformation followed by multi-class Linear Support Vector Machine (`LinearSVC`).
- **Score:** Raw score is SVM decision function $\in (-\infty, +\infty)$ (higher is better). Normalized similarity is mapped via logistic sigmoid $\frac{1}{1 + e^{-s}} \in [0, 1]$.

---

## 4. Empirical Benchmark Evaluation

Evaluation was conducted using a strict **leakage-safe split** with Session 1 gallery enrollment and Session 2 disjoint test probes. Zero overlap exists between enrollment, validation, and probe sets. Operating thresholds for 1:1 verification were tuned strictly on validation data prior to evaluating on test probes.

### 1:N Identification Accuracy (Rank-1 & Rank-5)
| Method | Enrollment Size | Rank-1 Accuracy | Rank-5 Accuracy | Extraction Latency | Match Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gabor + SVM** | 1 | 28.8% | 59.4% | 18.9 ms | 0.53 ms |
| **Gabor + SVM** | 3 | 36.4% | 66.0% | 47.3 ms | 1.39 ms |
| **Gabor + SVM** | 5 | 39.5% | 68.0% | 20.7 ms | 0.55 ms |
| **Gabor + SVM** | 10 | **46.7%** | **73.3%** | 102.7 ms | 2.06 ms |
| **LBP + KNN** | 1 | 37.9% | 73.6% | 9.9 ms | 0.15 ms |
| **LBP + KNN** | 3 | 38.8% | 76.4% | 10.5 ms | 0.30 ms |
| **LBP + KNN** | 5 | 37.0% | 72.6% | 35.6 ms | 0.74 ms |
| **LBP + KNN** | 10 | **43.8%** | **74.2%** | 45.2 ms | 4.57 ms |
| **Gabor + Hamming** | 1 | 27.5% | 57.0% | 27.8 ms | 29.8 ms |
| **Gabor + Hamming** | 3 | 28.4% | 53.7% | 28.3 ms | 82.9 ms |
| **Gabor + Hamming** | 5 | 31.0% | 58.1% | 17.7 ms | 276.1 ms |
| **Gabor + Hamming** | 10 | **30.4%** | **53.3%** | 20.9 ms | 296.3 ms |

### 1:1 Verification (Validation-Derived Thresholds)
| Method | Val Derived Threshold | Test FAR | Test FRR | Test EER |
|:---|:---:|:---:|:---:|:---:|
| **Gabor + Hamming** | 0.2503 | 9.00% | 72.29% | 41.30% |
| **LBP + KNN** | 0.3152 | 7.17% | 73.94% | 37.16% |
| **Gabor + SVM** | 0.9704 | 11.67% | 63.94% | 37.67% |

*Input Usability Rate:* 100% across all evaluated probes.

---

## 5. REST API Documentation (v1)

Base URL: `http://127.0.0.1:5000`

### `GET /api/v1/health`
Health check and enrolled gallery metadata.
```json
{
  "status": "healthy",
  "service": "ubipr-iris-recognition-api",
  "version": "1.0.0",
  "is_gallery_enrolled": true,
  "enrolled_subjects_count": 25,
  "available_methods": ["gabor_hamming", "lbp_knn", "gabor_svm"]
}
```

### `GET /api/v1/dataset/summary`
Returns the inventoried dataset audit details, session counts, and mask class semantics.

### `GET /api/v1/results/latest`
Returns the latest benchmark evaluation metrics and paths to generated charts.

### `POST /api/v1/identify`
1:N closed-set identification matching probe against all enrolled subjects.
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `image` (file, required): probe periocular image.
  - `mask` (file, optional): segmentation mask. If omitted, automated ocular detection runs.
  - `method` (text, optional): `gabor_hamming` | `lbp_knn` | `gabor_svm` | `all` (default `all`).
  - `top_k` (text, optional): number of candidates to return (default 5).

Example cURL:
```powershell
curl.exe -X POST "http://127.0.0.1:5000/api/v1/identify" `
  -F "image=@C:\Users\Piyush\OneDrive\Desktop\ubipr2\both_eyes\C101_S1_I1.jpg" `
  -F "method=all" `
  -F "top_k=3"
```

### `POST /api/v1/verify`
1:1 biometric verification against claimed identity using validation-derived threshold.
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `image` (file, required): probe periocular image.
  - `claimed_subject_id` (text, required): e.g. `C101`.
  - `method` (text, optional): `gabor_hamming` | `lbp_knn` | `gabor_svm`.

Example Response:
```json
{
  "status": "success",
  "claimed_subject_id": "C101",
  "decision": "MATCH",
  "is_verified_match": true,
  "method_name": "gabor_hamming",
  "raw_score": 0.2061,
  "score_type": "masked_hamming_distance",
  "higher_is_better": false,
  "normalized_similarity": 0.7939,
  "validation_threshold": 0.2503,
  "rank": 1,
  "quality": {
    "is_usable": true,
    "sharpness": 84.32
  }
}
```

---

## 6. Future Android & Spring Boot Integration Guide

The system is deliberately designed for seamless multi-tier expansion:
1. **Recognition Core Independence:** `src/services/recognition_service.py` accepts decoded numpy arrays or raw byte streams and returns typed python dictionaries. It has zero dependency on `flask.request`.
2. **Spring Boot Backend Gateway:** A Spring Boot application can accept uploads from mobile clients and forward multipart bytes via Spring `WebClient` or `RestTemplate` directly to `POST /api/v1/identify` and `POST /api/v1/verify`.
3. **Android Client (Kotlin / OkHttp / Retrofit):**
   - Use `MultipartBody.Part.createFormData("image", file.name, requestBody)`.
   - Never hardcode `localhost` or `127.0.0.1` inside Android code. For Android Emulator use `10.0.2.2:5000`; for physical test devices connected over Wi-Fi, bind Flask to `0.0.0.0` and configure the host's LAN IP (e.g. `192.168.1.x:5000`).

---

## 7. Quickstart & Verification (Windows PowerShell)

### Step 1: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Run Automated Test Suite
```powershell
python -m pytest -v
```
*(All 21 unit & integration tests pass with 100% coverage across labels, metrics, preprocessing, methods, and API)*

### Step 3: Run Dataset Audit
```powershell
python scripts/audit_dataset.py
```

### Step 4: Run Benchmark Evaluation
```powershell
python scripts/run_evaluation.py
```

### Step 5: Start Flask Web App
```powershell
python app.py
```
Open your browser to: `http://127.0.0.1:5000`
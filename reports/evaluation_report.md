# Comprehensive Iris Biometric Evaluation Report
## UBIPR Both-Eyes Benchmark

### 1. Benchmark Configuration
- **Dataset:** UBIPR Both-Eyes Visible Wavelength Dataset
- **Evaluated Methods:**
  1. `gabor_hamming`: Multi-Scale 2D Gabor Wavelet Bank + Masked Hamming Distance with Dual-Eye Rotation Compensation
  2. `lbp_knn`: Local Binary Patterns (LBP) Spatial Block Histograms + K-Nearest Neighbors
  3. `gabor_svm`: Multi-Filter Gabor Energy Statistics + Scaled Linear Support Vector Machine
- **Leakage Prevention Protocol:** Disjoint within-subject sample partition with Session 1 gallery and Session 2 probe prioritization. Zero overlap across enrollment, validation, and probe sets.
- **Derived Validation Thresholds:** Operating thresholds derived solely on validation sets prior to final test evaluation.

### 2. 1:N Identification Performance (Rank-1 & Rank-5 Accuracy)
| Method | Enrollment Size | Enrolled Subjects | Total Probes | Rank-1 Accuracy | Rank-5 Accuracy | Extr. Latency (ms) | Match Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gabor_hamming` | 1 | 25 | 465 | **27.5%** | 57.0% | 27.8 | 29.8 |
| `lbp_knn` | 1 | 25 | 465 | **37.9%** | 73.6% | 9.9 | 0.1 |
| `gabor_svm` | 1 | 25 | 465 | **28.8%** | 59.4% | 18.9 | 0.5 |
| `gabor_hamming` | 3 | 25 | 415 | **28.4%** | 53.7% | 28.2 | 82.9 |
| `lbp_knn` | 3 | 25 | 415 | **38.8%** | 76.4% | 10.4 | 0.3 |
| `gabor_svm` | 3 | 25 | 415 | **36.4%** | 66.0% | 47.3 | 1.4 |
| `gabor_hamming` | 5 | 25 | 365 | **31.0%** | 58.1% | 17.7 | 276.1 |
| `lbp_knn` | 5 | 25 | 365 | **37.0%** | 72.6% | 35.6 | 0.7 |
| `gabor_svm` | 5 | 25 | 365 | **39.5%** | 68.0% | 20.7 | 0.6 |
| `gabor_hamming` | 10 | 25 | 240 | **30.4%** | 53.3% | 20.9 | 296.3 |
| `lbp_knn` | 10 | 25 | 240 | **43.8%** | 74.2% | 45.2 | 4.6 |
| `gabor_svm` | 10 | 25 | 240 | **46.7%** | 73.3% | 102.7 | 2.1 |

### 3. 1:1 Verification Performance (Validation-Derived Operating Thresholds)
| Method | Val Derived Threshold | Test FAR | Test FRR | Test EER | Genuine Attempts | Impostor Attempts |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `gabor_hamming` | 0.2503 | **9.00%** | **72.29%** | **41.30%** | 3930 | 600 |
| `lbp_knn` | 0.3152 | **7.17%** | **73.94%** | **37.16%** | 3930 | 600 |
| `gabor_svm` | 0.9704 | **11.67%** | **63.94%** | **37.67%** | 3930 | 600 |

### 4. Generated Benchmark Figures
- Rank-1 Accuracy vs Enrollment: `outputs/figures/identification_accuracy.png`
- Verification ROC Curves: `outputs/figures/verification_roc.png`
- Computational Latency Breakdown: `outputs/figures/latency_comparison.png`

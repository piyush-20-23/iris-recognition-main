import csv
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    CHARTS_DIR,
    DEFAULT_RANDOM_SEED
)
from src.data.inventory import DatasetInventory
from src.evaluation.identification import run_identification_evaluation
from src.evaluation.protocols import create_leakage_safe_split
from src.evaluation.verification import run_verification_evaluation
from src.methods.gabor_hamming import GaborHammingMethod
from src.methods.lbp_knn import LbpKnnMethod
from src.methods.gabor_svm import GaborSvmMethod


def plot_and_save_charts(results: list, ver_results: list):
    """
    Generates PNG benchmark charts for the web frontend and evaluation report.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Identification Rank-1 vs Enrollment Size
    plt.figure(figsize=(8, 5))
    methods = sorted(list(set(r["method_name"] for r in results)))
    colors = {"gabor_hamming": "#1f77b4", "lbp_knn": "#2ca02c", "gabor_svm": "#d62728"}
    labels = {"gabor_hamming": "Gabor + Hamming", "lbp_knn": "LBP + KNN", "gabor_svm": "Gabor + SVM"}

    for m in methods:
        m_data = [r for r in results if r["method_name"] == m]
        m_data.sort(key=lambda x: x["enrollment_size"])
        xs = [r["enrollment_size"] for r in m_data]
        ys = [r["rank_1_accuracy"] * 100 for r in m_data]
        plt.plot(xs, ys, marker="o", linewidth=2.5, label=labels.get(m, m), color=colors.get(m, None))

    plt.title("Rank-1 Identification Accuracy vs. Enrollment Size", fontsize=13, fontweight="bold")
    plt.xlabel("Enrollment Gallery Size per Subject", fontsize=11)
    plt.ylabel("Rank-1 Accuracy (%)", fontsize=11)
    plt.xticks([1, 3, 5, 10])
    plt.ylim([0, 105])
    plt.legend(frameon=True)
    plt.tight_layout()

    fig1_path = FIGURES_DIR / "identification_accuracy.png"
    fig1_web = CHARTS_DIR / "identification_accuracy.png"
    plt.savefig(fig1_path, dpi=200)
    plt.savefig(fig1_web, dpi=200)
    plt.close()

    # 2. Verification ROC Curves (FAR vs FRR)
    plt.figure(figsize=(7, 5))
    for vr in ver_results:
        m = vr["method_name"]
        roc = vr.get("roc_curve", {})
        far = roc.get("far", [])
        frr = roc.get("frr", [])
        if far and frr:
            plt.plot(far, frr, linewidth=2, label=f"{labels.get(m, m)} (EER={vr['test_eer']*100:.1f}%)", color=colors.get(m, None))

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (EER=50%)")
    plt.title("Verification Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    plt.xlabel("False Accept Rate (FAR)", fontsize=11)
    plt.ylabel("False Reject Rate (FRR)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "verification_roc.png"
    fig2_web = CHARTS_DIR / "verification_roc.png"
    plt.savefig(fig2_path, dpi=200)
    plt.savefig(fig2_web, dpi=200)
    plt.close()

    # 3. Latency Comparison
    plt.figure(figsize=(8, 4.5))
    m_latencies = {}
    for m in methods:
        sub = [r for r in results if r["method_name"] == m]
        if sub:
            m_latencies[labels.get(m, m)] = (sub[0]["avg_extraction_latency_ms"], sub[0]["avg_matching_latency_ms"])

    x = np.arange(len(m_latencies))
    width = 0.35

    ext_times = [v[0] for v in m_latencies.values()]
    match_times = [v[1] for v in m_latencies.values()]

    plt.bar(x - width/2, ext_times, width, label="Feature Extraction (ms)", color="#3498db")
    plt.bar(x + width/2, match_times, width, label="1:N Match Latency (ms)", color="#e67e22")

    plt.ylabel("Latency (ms)", fontsize=11)
    plt.title("Processing Latency Comparison Across Biometric Pipelines", fontsize=13, fontweight="bold")
    plt.xticks(x, list(m_latencies.keys()))
    plt.legend()
    plt.tight_layout()

    fig3_path = FIGURES_DIR / "latency_comparison.png"
    fig3_web = CHARTS_DIR / "latency_comparison.png"
    plt.savefig(fig3_path, dpi=200)
    plt.savefig(fig3_web, dpi=200)
    plt.close()


def generate_evaluation_markdown(
    results: list,
    ver_results: list,
    output_path: Path
):
    md = """# Comprehensive Iris Biometric Evaluation Report
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
"""
    for r in results:
        md += (
            f"| `{r['method_name']}` | {r['enrollment_size']} | {r['enrolled_subjects_count']} | "
            f"{r['total_probes_evaluated']} | **{r['rank_1_accuracy']*100:.1f}%** | "
            f"{r['rank_5_accuracy']*100:.1f}% | {r['avg_extraction_latency_ms']:.1f} | "
            f"{r['avg_matching_latency_ms']:.1f} |\n"
        )

    md += """
### 3. 1:1 Verification Performance (Validation-Derived Operating Thresholds)
| Method | Val Derived Threshold | Test FAR | Test FRR | Test EER | Genuine Attempts | Impostor Attempts |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for vr in ver_results:
        md += (
            f"| `{vr['method_name']}` | {vr['validation_derived_threshold']:.4f} | "
            f"**{vr['test_far']*100:.2f}%** | **{vr['test_frr']*100:.2f}%** | "
            f"**{vr['test_eer']*100:.2f}%** | {vr['genuine_attempts']} | {vr['impostor_attempts']} |\n"
        )

    md += """
### 4. Generated Benchmark Figures
- Rank-1 Accuracy vs Enrollment: `outputs/figures/identification_accuracy.png`
- Verification ROC Curves: `outputs/figures/verification_roc.png`
- Computational Latency Breakdown: `outputs/figures/latency_comparison.png`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    print("=" * 60)
    print("RUNNING UBIPR BIOMETRIC BENCHMARK EVALUATION")
    print("=" * 60)

    inventory = DatasetInventory(DATASET_DIR)
    inventory.run_inventory(sample_decode_limit=50)

    # Enrollment sizes required by prompt: 1, 3, 5, 10
    enrollment_sizes = [1, 3, 5, 10]
    methods = [
        ("gabor_hamming", lambda: GaborHammingMethod()),
        ("lbp_knn", lambda: LbpKnnMethod()),
        ("gabor_svm", lambda: GaborSvmMethod())
    ]

    id_results = []
    ver_results = []

    # Use a cohort of 25 subjects for comprehensive yet fast benchmark execution
    max_cohort_subjects = 25

    print(f"\n[1/2] Evaluating 1:N Closed-Set Identification across enrollment sizes {enrollment_sizes}...")
    for enroll_size in enrollment_sizes:
        print(f"\n--- Protocol: Enrollment Size = {enroll_size} ---")
        split = create_leakage_safe_split(
            inventory.subject_samples,
            enrollment_size=enroll_size,
            val_samples_per_subject=2,
            max_subjects=max_cohort_subjects,
            session_aware=True,
            seed=DEFAULT_RANDOM_SEED
        )

        for m_name, m_factory in methods:
            m_instance = m_factory()
            res = run_identification_evaluation(m_instance, split, top_k=5)
            id_results.append(res)
            print(
                f"[{m_name}] Size={enroll_size} | "
                f"Rank-1 Acc: {res['rank_1_accuracy']*100:.1f}% | "
                f"Rank-5 Acc: {res['rank_5_accuracy']*100:.1f}% | "
                f"Extraction: {res['avg_extraction_latency_ms']} ms | "
                f"Match: {res['avg_matching_latency_ms']} ms"
            )

    print("\n[2/2] Evaluating 1:1 Verification (Validation-Derived Thresholds)...")
    ver_split = create_leakage_safe_split(
        inventory.subject_samples,
        enrollment_size=3,
        val_samples_per_subject=2,
        max_subjects=max_cohort_subjects,
        session_aware=True,
        seed=DEFAULT_RANDOM_SEED
    )

    for m_name, m_factory in methods:
        m_instance = m_factory()
        v_res = run_verification_evaluation(m_instance, ver_split, max_impostors=600)
        ver_results.append(v_res)
        print(
            f"[{m_name}] Val Thresh: {v_res['validation_derived_threshold']:.4f} | "
            f"Test FAR: {v_res['test_far']*100:.2f}% | "
            f"Test FRR: {v_res['test_frr']*100:.2f}% | "
            f"EER: {v_res['test_eer']*100:.2f}%"
        )

    # Export JSON
    summary_data = {
        "dataset_dir": str(DATASET_DIR),
        "cohort_subjects": max_cohort_subjects,
        "identification_results": id_results,
        "verification_results": [
            {k: v for k, v in vr.items() if k != "roc_curve"} for vr in ver_results
        ]
    }
    json_path = RESULTS_DIR / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Export CSV
    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "EnrollmentSize", "Rank1Accuracy", "Rank5Accuracy",
            "ExtractionLatencyMs", "MatchingLatencyMs", "TotalProbes"
        ])
        for r in id_results:
            writer.writerow([
                r["method_name"], r["enrollment_size"], r["rank_1_accuracy"],
                r["rank_5_accuracy"], r["avg_extraction_latency_ms"],
                r["avg_matching_latency_ms"], r["total_probes_evaluated"]
            ])

    # Generate charts
    print("\nGenerating evaluation charts...")
    plot_and_save_charts(id_results, ver_results)

    # Generate Markdown Report
    md_path = REPORTS_DIR / "evaluation_report.md"
    generate_evaluation_markdown(id_results, ver_results, md_path)

    print(f"\nSaved evaluation summary to: {json_path}")
    print(f"Saved evaluation CSV to: {csv_path}")
    print(f"Saved evaluation markdown report to: {md_path}")
    print("Evaluation benchmark complete!")


if __name__ == "__main__":
    main()

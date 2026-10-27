import json
import logging
import os
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from src.services.recognition_service import RecognitionService
from src.preprocessing.image_io import ImageReadError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("iris_app")

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = config.SECRET_KEY

# Initialize Recognition Service
recognition_service = RecognitionService(dataset_dir=config.DATASET_DIR)


@app.before_request
def start_timer():
    request.start_time = time.perf_counter()
    request.request_id = str(uuid.uuid4())


@app.after_request
def log_request(response):
    if hasattr(request, "start_time"):
        duration = (time.perf_counter() - request.start_time) * 1000.0
        logger.info(
            f"[{request.request_id[:8]}] {request.method} {request.path} "
            f"-> {response.status_code} ({duration:.2f}ms)"
        )
    return response


# --- Error Handlers (Return structured JSON for API, no stack traces) ---
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "status": "error",
        "error_type": "BadRequest",
        "message": str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error_type": "NotFound",
            "message": "The requested API endpoint was not found."
        }), 404
    return "Page Not Found", 404


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "error_type": "PayloadTooLarge",
        "message": f"File size exceeds maximum allowed size ({config.MAX_CONTENT_LENGTH // (1024*1024)} MB)."
    }), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    req_id = getattr(request, "request_id", "unknown")
    logger.error(f"[{req_id}] Unexpected error: {error}", exc_info=True)
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error_type": "InternalServerError",
            "request_id": req_id,
            "message": "An internal server error occurred while processing the request."
        }), 500
    return "Internal Server Error", 500


# --- Web UI Route ---
@app.route("/")
def index():
    return render_template("index.html")


# --- REST API v1 Routes ---

@app.route("/health", methods=["GET"])
@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "ubipr-iris-recognition-api",
        "version": "1.1.0",
        "timestamp": time.time(),
        "dataset_dir": str(config.DATASET_DIR),
        "is_gallery_enrolled": recognition_service.is_enrolled,
        "enrolled_subjects_count": len(recognition_service.enrolled_subjects),
        "enrolled_subjects": recognition_service.enrolled_subjects,
        "available_methods": list(recognition_service.methods.keys())
    })


@app.route("/api/v1/samples", methods=["GET"])
def list_samples():
    """Returns quick-testing sample probe images from enrolled subjects across the ID spectrum."""
    samples = []
    if config.DATASET_DIR.exists():
        # Pick sample images across various tiers: low (C1, C3), mid (C51, C99), high (C101, C201, C301, C401)
        target_subjs = ["C1", "C3", "C51", "C99", "C101", "C201", "C301", "C401"]
        for subj in target_subjs:
            if subj in recognition_service.enrolled_subjects:
                found = sorted(list(config.DATASET_DIR.glob(f"{subj}_*.jpg")))
                if found:
                    samples.append({
                        "subject_id": subj,
                        "filename": found[0].name,
                        "url": f"/api/v1/sample/{found[0].name}"
                    })
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/v1/sample/<filename>", methods=["GET"])
def get_sample(filename):
    """Serves a sample image file from the dataset for quick browser testing."""
    safe_filename = Path(filename).name
    if config.DATASET_DIR.exists():
        target = config.DATASET_DIR / safe_filename
        if target.exists() and target.is_file():
            return send_from_directory(config.DATASET_DIR, safe_filename)
    return jsonify({"status": "error", "message": "Sample image not found"}), 404


@app.route("/api/v1/dataset/summary", methods=["GET"])
def dataset_summary():
    json_path = config.REPORTS_DIR / "dataset_audit.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"status": "success", "summary": data})
    else:
        summary = recognition_service.inventory.run_inventory(sample_decode_limit=50)
        return jsonify({"status": "success", "summary": summary})


@app.route("/api/v1/results/latest", methods=["GET"])
def latest_results():
    results_path = config.RESULTS_DIR / "evaluation_summary.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({
            "status": "success",
            "data": data,
            "charts": [
                "/static/charts/identification_accuracy.png",
                "/static/charts/verification_roc.png",
                "/static/charts/latency_comparison.png"
            ]
        })
    return jsonify({
        "status": "error",
        "message": "No evaluation benchmark results found. Run scripts/run_evaluation.py first."
    }), 404


def _ensure_gallery_enrolled() -> bool:
    if recognition_service.is_enrolled:
        return True
    if config.DATASET_DIR.exists() and any(config.DATASET_DIR.glob("*.jpg")):
        try:
            logger.info("Auto-enrolling gallery on first request...")
            recognition_service.initialize_and_enroll(enrollment_size=3, max_subjects=50)
            return recognition_service.is_enrolled
        except Exception as e:
            logger.error(f"Failed to auto-enroll gallery: {e}")
            return False
    return False


@app.route("/api/v1/identify", methods=["POST"])
def identify():
    req_id = getattr(request, "request_id", str(uuid.uuid4()))

    if not _ensure_gallery_enrolled():
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "error_type": "GalleryNotEnrolled",
            "message": "Gallery has not been enrolled. The UBIPr dataset is required to enroll gallery subjects for matching. Please configure the dataset path or download the dataset."
        }), 503

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Missing required multipart field 'image'."
        }), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Empty or missing image filename."
        }), 400

    image_bytes = image_file.read()
    if len(image_bytes) == 0:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Uploaded image file contains zero bytes."
        }), 400

    mask_bytes = None
    if "mask" in request.files:
        m_file = request.files["mask"]
        if m_file and m_file.filename != "":
            mask_bytes = m_file.read()

    # Auto-resolve mask from dataset if not explicitly uploaded
    if mask_bytes is None and image_file.filename:
        stem = Path(image_file.filename).stem
        gt_mask_path = config.DATASET_DIR / f"{stem}.png"
        if gt_mask_path.exists():
            try:
                mask_bytes = gt_mask_path.read_bytes()
                logger.info(f"Auto-resolved ground-truth segmentation mask for '{image_file.filename}'.")
            except Exception as e:
                logger.warning(f"Could not read auto-resolved mask: {e}")

    method_name = request.form.get("method", "all").strip().lower()
    top_k_str = request.form.get("top_k", "5")
    try:
        top_k = max(1, min(50, int(top_k_str)))
    except ValueError:
        top_k = 5

    try:
        result = recognition_service.identify(
            image_source=image_bytes,
            mask_source=mask_bytes,
            method_name=method_name,
            top_k=top_k
        )
        result["request_id"] = req_id
        return jsonify(result), 200

    except ImageReadError as e:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "error_type": "ImageReadError",
            "message": str(e)
        }), 400
    except ValueError as e:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "error_type": "InvalidParameter",
            "message": str(e)
        }), 400


@app.route("/api/v1/verify", methods=["POST"])
def verify():
    req_id = getattr(request, "request_id", str(uuid.uuid4()))

    if not _ensure_gallery_enrolled():
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "error_type": "GalleryNotEnrolled",
            "message": "Gallery has not been enrolled. The UBIPr dataset is required to verify subjects. Please configure the dataset path or download the dataset."
        }), 503

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Missing required multipart field 'image'."
        }), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Empty or missing image filename."
        }), 400

    claimed_subject_id = request.form.get("claimed_subject_id", "").strip()
    if not claimed_subject_id:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Missing required form field 'claimed_subject_id'."
        }), 400

    image_bytes = image_file.read()
    if len(image_bytes) == 0:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "message": "Uploaded image file contains zero bytes."
        }), 400

    mask_bytes = None
    if "mask" in request.files:
        m_file = request.files["mask"]
        if m_file and m_file.filename != "":
            mask_bytes = m_file.read()

    # Auto-resolve mask from dataset if not explicitly uploaded
    if mask_bytes is None and image_file.filename:
        stem = Path(image_file.filename).stem
        gt_mask_path = config.DATASET_DIR / f"{stem}.png"
        if gt_mask_path.exists():
            try:
                mask_bytes = gt_mask_path.read_bytes()
                logger.info(f"Auto-resolved ground-truth segmentation mask for '{image_file.filename}'.")
            except Exception as e:
                logger.warning(f"Could not read auto-resolved mask: {e}")

    method_name = request.form.get("method", "ensemble_fusion").strip().lower()

    try:
        result = recognition_service.verify(
            image_source=image_bytes,
            claimed_subject_id=claimed_subject_id,
            mask_source=mask_bytes,
            method_name=method_name
        )
        result["request_id"] = req_id
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), status_code

    except ImageReadError as e:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "error_type": "ImageReadError",
            "message": str(e)
        }), 400
    except ValueError as e:
        return jsonify({
            "status": "error",
            "request_id": req_id,
            "error_type": "InvalidParameter",
            "message": str(e)
        }), 400


if __name__ == "__main__":
    if config.DATASET_DIR.exists():
        logger.info(f"Pre-warming recognition service gallery models from '{config.DATASET_DIR}'...")
        try:
            recognition_service.initialize_and_enroll(enrollment_size=3, max_subjects=None)
            logger.info(f"Gallery pre-warming completed successfully ({len(recognition_service.enrolled_subjects)} subjects).")
        except Exception as e:
            logger.warning(f"Could not pre-warm gallery: {e}. Starting server with un-enrolled gallery.")
    else:
        logger.warning(
            f"Dataset directory not found at '{config.DATASET_DIR}'. "
            "Server starting in dashboard mode without enrolled gallery. "
            "Download the UBIPr both_eyes dataset to enable live 1:N identification and 1:1 verification."
        )
    logger.info(f"Starting Flask server on http://{config.FLASK_HOST}:{config.FLASK_PORT}...")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)

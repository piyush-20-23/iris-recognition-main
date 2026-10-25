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

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "ubipr-iris-recognition-api",
        "version": "1.0.0",
        "timestamp": time.time(),
        "dataset_dir": str(config.DATASET_DIR),
        "is_gallery_enrolled": recognition_service.is_enrolled,
        "enrolled_subjects_count": len(recognition_service.enrolled_subjects),
        "enrolled_subjects": recognition_service.enrolled_subjects[:15],
        "available_methods": list(recognition_service.methods.keys())
    })


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


@app.route("/api/v1/identify", methods=["POST"])
def identify():
    req_id = getattr(request, "request_id", str(uuid.uuid4()))

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

    method_name = request.form.get("method", "gabor_hamming").strip().lower()

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
    logger.info("Pre-warming recognition service gallery models...")
    recognition_service.initialize_and_enroll(enrollment_size=3, max_subjects=25)
    logger.info(f"Starting Flask server on http://{config.FLASK_HOST}:{config.FLASK_PORT}...")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)

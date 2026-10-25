import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Dataset path: environment variable takes precedence, fallback to default local desktop location
DEFAULT_DATASET_DIR = Path(r"c:\Users\Piyush\OneDrive\Desktop\ubipr2\both_eyes")
DATASET_DIR = Path(os.environ.get("IRIS_DATASET_DIR", str(DEFAULT_DATASET_DIR)))

# Output directories
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TEMPLATES_DIR = ARTIFACTS_DIR / "templates"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
STATIC_DIR = BASE_DIR / "static"
CHARTS_DIR = STATIC_DIR / "charts"

# Ensure runtime directories exist
for p in [REPORTS_DIR, ARTIFACTS_DIR, MODELS_DIR, TEMPLATES_DIR, OUTPUTS_DIR, RESULTS_DIR, FIGURES_DIR, STATIC_DIR, CHARTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset specifications
IGNORE_PATTERNS = {".DS_Store", "Thumbs.db"}
IGNORE_PREFIXES = ("._", ".")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# UBIPR ground truth mask pixel class mappings
MASK_CLASS_BG_SKIN = 0
MASK_CLASS_IRIS = 85
MASK_CLASS_SCLERA = 170
MASK_CLASS_EYEBROWS = 255

# Normalization standards
NORMALIZED_HEIGHT = 64
NORMALIZED_WIDTH_PER_EYE = 256
NORMALIZED_WIDTH_COMBINED = 512

# Methods & hyperparameter defaults
DEFAULT_RANDOM_SEED = 42
DEFAULT_ENROLLMENT_SIZES = [1, 3, 5, 10]
DEFAULT_KNN_NEIGHBORS = 3
DEFAULT_KNN_METRIC = "euclidean"
DEFAULT_SVM_KERNEL = "linear"
DEFAULT_SVM_C = 1.0

# Gabor Iris Code parameters
GABOR_NUM_FILTERS = 4
GABOR_FREQUENCIES = [0.1, 0.2]

# Flask App Configuration
FLASK_HOST = os.environ.get("IRIS_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("IRIS_PORT", 5000))
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload
SECRET_KEY = os.environ.get("IRIS_SECRET_KEY", "iris-biometric-secret-key-2026")

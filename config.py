from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "face_model.pkl"

IMAGE_SIZE = (64, 64)
CAPTURE_SAMPLES = 7
MIN_USERS = 2
MIN_SAMPLES_FOR_TRAINING = 2
RECOMMENDED_SAMPLES_PER_USER = 5
TRAIN_TEST_SIZE = 0.30
RANDOM_STATE = 42
KNN_NEIGHBORS = 3
PROBABILITY_THRESHOLD = 0.55
DISTANCE_THRESHOLD_MULTIPLIER = 1.15
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_directories() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    MIN_SAMPLES_FOR_TRAINING,
    MIN_USERS,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RECOMMENDED_SAMPLES_PER_USER,
    SUPPORTED_IMAGE_SUFFIXES,
    ensure_directories,
)
from src.data_collection import save_uploaded_face_image, slugify_user_id  # noqa: E402
from src.predict import load_model_artifact, predict_uploaded_image  # noqa: E402
from src.preprocessing import preprocess_image_file  # noqa: E402
from src.train import train_face_model  # noqa: E402


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def get_project_status() -> dict:
    ensure_directories()
    user_ids = sorted(
        {path.name for path in RAW_DATA_DIR.iterdir() if path.is_dir()}
        | {path.name for path in PROCESSED_DATA_DIR.iterdir() if path.is_dir()}
    )

    users = []
    for user_id in user_ids:
        raw_count = count_images(RAW_DATA_DIR / user_id)
        processed_count = count_images(PROCESSED_DATA_DIR / user_id)
        users.append(
            {
                "user_id": user_id,
                "raw_samples": raw_count,
                "processed_samples": processed_count,
                "ready_for_training": processed_count >= MIN_SAMPLES_FOR_TRAINING,
                "recommended_complete": processed_count >= RECOMMENDED_SAMPLES_PER_USER,
            }
        )

    metrics = {}
    if MODEL_PATH.exists():
        try:
            metrics = load_model_artifact().get("metrics", {})
        except Exception:
            metrics = {}

    return {
        "users": users,
        "registered_user_count": len(users),
        "trainable_users": sum(1 for user in users if user["ready_for_training"]),
        "model_ready": MODEL_PATH.exists(),
        "metrics": metrics,
    }


def register_face_sample(user_name: str, image_bytes: bytes) -> dict:
    ensure_directories()
    user_id = slugify_user_id(user_name)
    raw_path = save_uploaded_face_image(user_id, image_bytes)
    processed_path = PROCESSED_DATA_DIR / user_id / f"{raw_path.stem}.png"
    preprocess_image_file(raw_path, output_path=processed_path)

    processed_count = count_images(PROCESSED_DATA_DIR / user_id)
    return {
        "user_id": user_id,
        "raw_path": raw_path,
        "processed_path": processed_path,
        "processed_count": processed_count,
        "remaining_recommended_samples": max(0, RECOMMENDED_SAMPLES_PER_USER - processed_count),
    }


def train_model_from_registered_faces() -> dict:
    return train_face_model()


def authenticate_uploaded_face(image_bytes: bytes) -> dict:
    return predict_uploaded_image(image_bytes)


def training_readiness_message() -> str:
    status = get_project_status()
    if status["trainable_users"] < MIN_USERS:
        return f"Register at least {MIN_USERS} users with {MIN_SAMPLES_FOR_TRAINING}+ processed images each before training."
    return "The dataset is ready for training."

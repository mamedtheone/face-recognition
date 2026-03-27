from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import (
    CASCADE_FILENAME,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SUPPORTED_IMAGE_SUFFIXES,
    ensure_directories,
)


def build_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FILENAME)
    if detector.empty():
        raise FileNotFoundError("OpenCV Haar Cascade file could not be loaded.")
    return detector


def iter_image_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def detect_largest_face(
    image: np.ndarray,
    detector: cv2.CascadeClassifier | None = None,
) -> tuple[int, int, int, int] | None:
    detector = detector or build_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


def preprocess_image(
    image: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
    detector: cv2.CascadeClassifier | None = None,
) -> np.ndarray:
    if image is None:
        raise ValueError("Image is empty.")

    detector = detector or build_face_detector()
    box = detect_largest_face(image, detector=detector)

    if box is not None:
        x, y, w, h = box
        image = image[y : y + h, x : x + w]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def save_processed_image(processed_image: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_to_save = np.clip(processed_image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), image_to_save)
    return output_path


def preprocess_image_file(
    image_path: Path,
    output_path: Path | None = None,
    detector: cv2.CascadeClassifier | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    processed = preprocess_image(image, detector=detector)
    if output_path is not None:
        save_processed_image(processed, output_path)
    return processed


def preprocess_user_directory(
    user_id: str,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> int:
    ensure_directories()
    detector = build_face_detector()
    source_dir = raw_dir / user_id
    target_dir = processed_dir / user_id
    target_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for image_path in iter_image_files(source_dir):
        output_path = target_dir / f"{image_path.stem}.png"
        preprocess_image_file(image_path, output_path=output_path, detector=detector)
        processed_count += 1

    return processed_count


def preprocess_dataset(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR) -> int:
    ensure_directories()
    total_processed = 0
    for user_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        total_processed += preprocess_user_directory(user_dir.name, raw_dir=raw_dir, processed_dir=processed_dir)
    return total_processed

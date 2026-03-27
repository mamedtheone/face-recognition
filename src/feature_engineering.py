from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import PROCESSED_DATA_DIR, SUPPORTED_IMAGE_SUFFIXES


def flatten_image(processed_image: np.ndarray) -> np.ndarray:
    return processed_image.astype(np.float32).reshape(-1)


def load_processed_dataset(processed_dir: Path = PROCESSED_DATA_DIR) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    if not processed_dir.exists():
        raise ValueError("Processed dataset directory does not exist.")

    for user_dir in sorted(path for path in processed_dir.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in user_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            processed_image = image.astype(np.float32)
            if processed_image.max() > 1.0:
                processed_image /= 255.0
            features.append(flatten_image(processed_image))
            labels.append(user_dir.name)

    if not features:
        raise ValueError("No processed face images were found. Register users before training.")

    return np.vstack(features), np.array(labels), dict(Counter(labels))

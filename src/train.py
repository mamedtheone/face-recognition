from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DISTANCE_THRESHOLD_MULTIPLIER,
    KNN_NEIGHBORS,
    MIN_SAMPLES_FOR_TRAINING,
    MIN_USERS,
    MODEL_PATH,
    PROBABILITY_THRESHOLD,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TRAIN_TEST_SIZE,
    ensure_directories,
)
from src.evaluate import evaluate_predictions
from src.feature_engineering import load_processed_dataset
from src.preprocessing import preprocess_dataset


def validate_dataset(class_counts: dict[str, int]) -> None:
    if len(class_counts) < MIN_USERS:
        raise ValueError(f"At least {MIN_USERS} users are required for training.")

    insufficient = [label for label, count in class_counts.items() if count < MIN_SAMPLES_FOR_TRAINING]
    if insufficient:
        names = ", ".join(sorted(insufficient))
        raise ValueError(
            f"Each user needs at least {MIN_SAMPLES_FOR_TRAINING} processed images for training. Missing: {names}"
        )


def build_classifier(n_neighbors: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        ]
    )


def resolve_test_size(class_counts: dict[str, int]) -> int:
    total_samples = sum(class_counts.values())
    class_count = len(class_counts)
    requested_test_samples = max(class_count, int(round(total_samples * TRAIN_TEST_SIZE)))
    max_allowed_test_samples = total_samples - class_count
    if max_allowed_test_samples < class_count:
        raise ValueError("Not enough samples to create stratified train and test splits.")
    return min(requested_test_samples, max_allowed_test_samples)


def calibrate_distance_threshold(model: Pipeline, X_train: np.ndarray) -> float:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    transformed = scaler.transform(X_train)
    neighbor_count = min(classifier.n_neighbors + 1, len(transformed))
    distances, _ = classifier.kneighbors(transformed, n_neighbors=neighbor_count)
    if distances.shape[1] > 1:
        distances = distances[:, 1:]
    return float(np.percentile(distances.mean(axis=1), 95) * DISTANCE_THRESHOLD_MULTIPLIER)


def train_face_model(
    processed_dir: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODEL_PATH,
    auto_preprocess: bool = True,
) -> dict:
    ensure_directories()
    if auto_preprocess:
        preprocess_dataset(processed_dir=processed_dir)

    X, y, class_counts = load_processed_dataset(processed_dir)
    validate_dataset(class_counts)
    test_size = resolve_test_size(class_counts)

    n_neighbors = max(1, min(KNN_NEIGHBORS, min(class_counts.values())))
    model = build_classifier(n_neighbors=n_neighbors)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = sorted(class_counts)
    metrics = evaluate_predictions(y_test, y_pred, labels=labels)
    distance_threshold = calibrate_distance_threshold(model, X_train)

    artifact = {
        "model": model,
        "metrics": metrics,
        "class_counts": class_counts,
        "image_size": X.shape[1],
        "distance_threshold": distance_threshold,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FaceAuth KNN model.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    artifact = train_face_model(
        processed_dir=args.processed_dir,
        model_path=args.model_path,
        auto_preprocess=not args.skip_preprocess,
    )
    print(f"Training complete. Accuracy: {artifact['metrics']['accuracy']:.3f}")


if __name__ == "__main__":
    main()

from pathlib import Path

import cv2
import numpy as np

from src.predict import predict_image_file
from src.preprocessing import preprocess_image
from src.train import train_face_model


def _write_synthetic_samples(user_dir: Path, background: int, center: int) -> None:
    user_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        image = np.full((80, 80, 3), background, dtype=np.uint8)
        cv2.circle(image, (40, 40), 18, (center + index, center + index, center + index), -1)
        cv2.imwrite(str(user_dir / f"sample_{index}.png"), image)


def test_preprocess_image_returns_normalized_64x64_output():
    image = np.full((90, 90, 3), 140, dtype=np.uint8)
    processed = preprocess_image(image)
    assert processed.shape == (64, 64)
    assert 0.0 <= float(processed.min()) <= float(processed.max()) <= 1.0


def test_training_and_prediction_pipeline(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    model_path = tmp_path / "face_model.pkl"

    _write_synthetic_samples(processed_dir / "alice", background=30, center=220)
    _write_synthetic_samples(processed_dir / "bob", background=220, center=30)

    artifact = train_face_model(processed_dir=processed_dir, model_path=model_path, auto_preprocess=False)

    assert model_path.exists()
    assert artifact["metrics"]["accuracy"] >= 0.50

    result = predict_image_file(processed_dir / "alice" / "sample_0.png", model_path=model_path)
    assert result["predicted_label"] == "alice"

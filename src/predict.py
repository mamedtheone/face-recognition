from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from config import MODEL_PATH, PROBABILITY_THRESHOLD
from src.preprocessing import preprocess_image


def load_model_artifact(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Train the model first.")
    with model_path.open("rb") as file:
        return pickle.load(file)


def compute_average_distance(model, features: np.ndarray) -> float:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    transformed = scaler.transform([features])
    neighbor_count = min(classifier.n_neighbors, len(classifier._fit_X))
    distances, _ = classifier.kneighbors(transformed, n_neighbors=neighbor_count)
    return float(distances.mean())


def predict_face(image: np.ndarray, model_path: Path = MODEL_PATH) -> dict:
    artifact = load_model_artifact(model_path)
    model = artifact["model"]
    processed = preprocess_image(image)
    features = processed.reshape(1, -1)

    predicted_label = str(model.predict(features)[0])
    probability = 1.0
    if hasattr(model, "predict_proba"):
        probability = float(np.max(model.predict_proba(features)[0]))

    distance = compute_average_distance(model, features[0])
    distance_threshold = float(artifact.get("distance_threshold", float("inf")))
    probability_threshold = float(artifact.get("probability_threshold", PROBABILITY_THRESHOLD))
    granted = distance <= distance_threshold and probability >= probability_threshold

    return {
        "predicted_label": predicted_label,
        "probability": probability,
        "distance": distance,
        "distance_threshold": distance_threshold,
        "granted": granted,
        "message": f"Access Granted: {predicted_label}" if granted else "Access Denied",
    }


def predict_image_file(image_path: Path, model_path: Path = MODEL_PATH) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return predict_face(image, model_path=model_path)


def predict_uploaded_image(image_bytes: bytes, model_path: Path = MODEL_PATH) -> dict:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded image could not be decoded.")
    return predict_face(image, model_path=model_path)


def capture_and_predict(camera_index: int = 0, model_path: Path = MODEL_PATH) -> dict:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue
            cv2.putText(
                frame,
                "Press SPACE to authenticate, Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("FaceAuth Login", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                return predict_face(frame, model_path=model_path)
            if key == ord("q"):
                raise KeyboardInterrupt("Login cancelled by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a face from an image file.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    result = predict_image_file(args.image_path, model_path=args.model_path)
    print(result["message"])
    print(f"Predicted label: {result['predicted_label']}")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Distance: {result['distance']:.3f}")


if __name__ == "__main__":
    main()

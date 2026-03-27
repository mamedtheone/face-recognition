from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import MODEL_PATH


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def load_saved_metrics(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError("Model file not found.")
    with model_path.open("rb") as file:
        artifact = pickle.load(file)
    return artifact.get("metrics", {})


def main() -> None:
    metrics = load_saved_metrics()
    print(f"Accuracy: {metrics.get('accuracy', 0.0):.3f}")
    print("Confusion matrix:")
    for row in metrics.get("confusion_matrix", []):
        print(row)


if __name__ == "__main__":
    main()

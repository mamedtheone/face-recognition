from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from config import CAPTURE_SAMPLES, RAW_DATA_DIR, ensure_directories


def slugify_user_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "user"


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the uploaded image.")
    return image


def save_uploaded_face_image(user_id: str, image_bytes: bytes, base_dir: Path = RAW_DATA_DIR) -> Path:
    ensure_directories()
    user_dir = base_dir / slugify_user_id(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"sample_{int(time.time() * 1000)}.jpg"
    image = decode_uploaded_image(image_bytes)
    cv2.imwrite(str(file_path), image)
    return file_path


def capture_face_images(
    user_id: str,
    num_samples: int = CAPTURE_SAMPLES,
    camera_index: int = 0,
    base_dir: Path = RAW_DATA_DIR,
) -> list[Path]:
    ensure_directories()
    normalized_user_id = slugify_user_id(user_id)
    user_dir = base_dir / normalized_user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        camera.release()
        raise RuntimeError("OpenCV Haar Cascade could not be loaded.")

    saved_images: list[Path] = []
    last_capture_time = 0.0

    try:
        while len(saved_images) < num_samples:
            success, frame = camera.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Samples: {len(saved_images)}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press Q to cancel",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if len(faces) > 0 and time.time() - last_capture_time > 0.5:
                file_path = user_dir / f"sample_{len(saved_images) + 1:02d}.jpg"
                cv2.imwrite(str(file_path), frame)
                saved_images.append(file_path)
                last_capture_time = time.time()

            cv2.imshow("FaceAuth Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return saved_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture face samples from a webcam.")
    parser.add_argument("user_id", help="Name or ID for the user.")
    parser.add_argument("--samples", type=int, default=CAPTURE_SAMPLES, help="Number of images to capture.")
    args = parser.parse_args()
    saved = capture_face_images(args.user_id, num_samples=args.samples)
    print(f"Saved {len(saved)} images for {slugify_user_id(args.user_id)}")


if __name__ == "__main__":
    main()

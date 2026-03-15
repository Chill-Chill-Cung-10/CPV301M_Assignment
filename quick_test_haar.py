from pathlib import Path
import argparse

import cv2
import numpy as np


DATASET_ROOT = Path("datasource")
DEFAULT_OUTPUT = Path(r"outputs\haar_face_test.jpg")


def find_first_image(dataset_root: Path) -> Path:
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        match = next(dataset_root.rglob(pattern), None)
        if match is not None:
            return match
    raise FileNotFoundError(f"No image found under: {dataset_root}")


def detect_and_draw(image_path: Path, output_path: Path) -> int:
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Cannot load Haar Cascade: {cascade_path}")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix or ".jpg", image)
    if not success:
        raise RuntimeError(f"Cannot encode output image: {output_path}")
    encoded.tofile(str(output_path))
    return len(faces)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick Haar Cascade face detection test on one dataset image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to an input image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the boxed output image.",
    )
    args = parser.parse_args()

    image_path = args.image or find_first_image(DATASET_ROOT)
    count = detect_and_draw(image_path, args.output)
    print(f"Input image : {image_path}")
    print(f"Output image: {args.output}")
    print(f"Faces found : {count}")


if __name__ == "__main__":
    main()

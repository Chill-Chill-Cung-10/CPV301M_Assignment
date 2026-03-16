from pathlib import Path
from PIL import Image
from urllib.request import urlretrieve
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import ResNet18_Weights, resnet18

DATASET_ROOT = Path("datasource")
CACHE_FILE = Path("datasource_resnet_yunet_features.npz")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FACE_SIZE = (160, 160)
MODEL_INPUT_SIZE = (224, 224)
GALLERY_IMAGES_PER_PERSON = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YUNET_MODEL_PATH = Path("face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
 )


def ensure_file(file_path, url, min_size_bytes=10000):
    """Download a model file once if it is missing or clearly invalid."""
    if file_path.exists() and file_path.stat().st_size >= min_size_bytes:
        return file_path

    if file_path.exists():
        print(f"Replacing invalid file: {file_path.name}")
        file_path.unlink()

    print(f"Downloading {file_path.name} ...")
    urlretrieve(url, str(file_path))

    if file_path.stat().st_size < min_size_bytes:
        raise RuntimeError(f"Downloaded file looks invalid: {file_path}")

    return file_path


def load_embedding_model():
    """Load a pretrained CNN and use its penultimate layer as face embedding."""
    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained ResNet18 weights. Internet may be required on the first run."
        ) from exc

    model.fc = torch.nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model, weights.transforms()


def load_yunet_detector():
    """Load YuNet face detector, using CUDA backend when OpenCV supports it."""
    ensure_file(YUNET_MODEL_PATH, YUNET_MODEL_URL)

    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = cv2.dnn.DNN_TARGET_CPU
    detector_mode = "CPU"

    if DEVICE.type == "cuda":
        try:
            detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH),
                "",
                (320, 320),
                0.85,
                0.3,
                5000,
                cv2.dnn.DNN_BACKEND_CUDA,
                cv2.dnn.DNN_TARGET_CUDA,
            )
            detector_mode = "CUDA"
            return detector, detector_mode
        except cv2.error:
            detector_mode = "CPU fallback"

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        0.85,
        0.3,
        5000,
        backend_id,
        target_id,
    )
    return detector, detector_mode


embedding_model, embedding_preprocess = load_embedding_model()
face_detector, detector_mode = load_yunet_detector()
print(f"Embedding model ready on {DEVICE}.")
print(f"Face detector ready with YuNet on {detector_mode}.")


def detect_largest_face(image_bgr):
    """Return (x, y, w, h) for the largest detected face, or None."""
    height, width = image_bgr.shape[:2]
    face_detector.setInputSize((width, height))
    _, detections = face_detector.detect(image_bgr)
    if detections is None or len(detections) == 0:
        return None

    best = max(detections, key=lambda det: det[2] * det[3])
    x, y, w, h = best[:4]
    return int(x), int(y), int(w), int(h)


def expand_face_box(box, image_shape, scale=0.2):
    """Expand the detected box a bit to keep more facial context."""
    x, y, w, h = box
    pad_w = int(w * scale)
    pad_h = int(h * scale)
    h_img, w_img = image_shape[:2]

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    return x1, y1, x2 - x1, y2 - y1


def extract_face_feature_from_bgr(image_bgr):
    """Extract a normalized deep embedding from the detected face crop."""
    box = detect_largest_face(image_bgr)
    if box is None:
        return None

    x, y, w, h = expand_face_box(box, image_bgr.shape, scale=0.2)
    face_bgr = image_bgr[y:y+h, x:x+w]
    if face_bgr.size == 0:
        return None

    face_bgr = cv2.resize(face_bgr, FACE_SIZE, interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    face_image = Image.fromarray(face_rgb)
    input_tensor = embedding_preprocess(face_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        embedding = embedding_model(input_tensor).squeeze(0).cpu().numpy().astype(np.float32)

    embedding /= np.linalg.norm(embedding) + 1e-8
    return embedding


def read_image_bgr(image_path):
    """Read image paths safely on Windows when paths contain non-ASCII chars."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_identity_dirs(dataset_root):
    """Find leaf identity directories that contain image files directly."""
    identity_dirs = []
    for directory in dataset_root.rglob("*"):
        if not directory.is_dir():
            continue

        has_images = any(
            p.is_file() and p.suffix.lower() in VALID_EXTS
            for p in directory.iterdir()
        )
        if has_images:
            identity_dirs.append(directory)

    identity_dirs.sort(key=lambda p: str(p.relative_to(dataset_root)).lower())
    return identity_dirs


def build_or_load_gallery_features(dataset_root, cache_file, max_images_per_person=GALLERY_IMAGES_PER_PERSON):
    """Build deep face embeddings once and cache for faster reruns."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        features = data["features"]
        image_paths = data["image_paths"].tolist()
        labels = data["labels"].tolist()
        print(f"Loaded cache: {len(features)} samples")
        return features, image_paths, labels

    features = []
    image_paths = []
    labels = []

    person_dirs = find_identity_dirs(dataset_root)
    if not person_dirs:
        raise RuntimeError("No identity folders with image files found in dataset.")

    for person_dir in person_dirs:
        images = sorted(
            [
                p for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            ],
            key=lambda p: p.name.lower(),
        )[:max_images_per_person]

        for img_path in images:
            img_bgr = read_image_bgr(img_path)
            if img_bgr is None:
                continue

            feat = extract_face_feature_from_bgr(img_bgr)
            if feat is None:
                continue

            features.append(feat)
            image_paths.append(str(img_path))
            labels.append(str(person_dir.relative_to(dataset_root)))

    if not features:
        raise RuntimeError("No valid face features found in dataset.")

    features = np.asarray(features, dtype=np.float32)
    np.savez_compressed(
        cache_file,
        features=features,
        image_paths=np.array(image_paths, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"Built and cached: {len(features)} samples")
    return features, image_paths, labels


def capture_single_frame_from_webcam():
    """Press SPACE to capture one frame, ESC to cancel."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    captured = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        preview = frame.copy()
        box = detect_largest_face(preview)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            preview,
            f"SPACE: capture | ESC: exit | embed={DEVICE.type} | detect={detector_mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture Face", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            captured = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def cosine_similarity_matrix(query, gallery):
    """Cosine similarities between query vector and gallery matrix."""
    query_norm = np.linalg.norm(query) + 1e-8
    gallery_norms = np.linalg.norm(gallery, axis=1) + 1e-8
    return (gallery @ query) / (gallery_norms * query_norm)


# 1) Build/load gallery features (first run may be slower than histogram)
gallery_features, gallery_paths, gallery_labels = build_or_load_gallery_features(
    DATASET_ROOT,
    CACHE_FILE,
    max_images_per_person=GALLERY_IMAGES_PER_PERSON,
)

# 2) Capture webcam image
query_bgr = capture_single_frame_from_webcam()
if query_bgr is None:
    raise RuntimeError("No image captured.")

# 3) Extract query feature
query_feature = extract_face_feature_from_bgr(query_bgr)
if query_feature is None:
    raise RuntimeError("No face detected in captured image.")

# 4) Find most similar image
scores = cosine_similarity_matrix(query_feature, gallery_features)
best_idx = int(np.argmax(scores))
best_score = float(scores[best_idx])
best_path = gallery_paths[best_idx]
best_label = gallery_labels[best_idx]

print(f"Best match folder: {best_label}")
print(f"Best match image: {best_path}")
print(f"Similarity score: {best_score:.4f}")

# 5) Display query and best match side-by-side
best_bgr = read_image_bgr(best_path)
if best_bgr is None:
    raise RuntimeError("Failed to load best-match image.")

query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
best_rgb = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(query_rgb)
plt.title("Captured image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(best_rgb)
plt.title(f"Best match (ID={best_label})\nScore={best_score:.4f}")
plt.axis("off")

plt.tight_layout()
plt.show()
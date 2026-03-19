from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import onnxruntime as ort
import torch

DATASET_ROOT = Path("datasource_v2")
CACHE_FILE = Path("datasource_v2_arcface_features.npz")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FACE_ALIGNMENT_SIZE = (112, 112)
GALLERY_IMAGES_PER_PERSON = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YUNET_MODEL_PATH = Path("face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

ARCFACE_MODEL_PATH = Path("arcfaceresnet100-8.onnx")
ARCFACE_MODEL_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
    "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx",
]


def ensure_file(file_path, url_or_urls, min_size_bytes=10000):
    if file_path.exists() and file_path.stat().st_size >= min_size_bytes:
        return file_path

    urls = url_or_urls if isinstance(url_or_urls, (list, tuple)) else [url_or_urls]
    last_error = None

    for url in urls:
        try:
            if file_path.exists():
                file_path.unlink()

            print(f"Downloading {file_path.name} from: {url}")
            urlretrieve(url, str(file_path))

            if file_path.stat().st_size < min_size_bytes:
                raise RuntimeError(
                    f"Downloaded file too small ({file_path.stat().st_size} bytes)"
                )

            print(f"Downloaded {file_path.name} ({file_path.stat().st_size // 1024} KB)")
            return file_path
        except Exception as exc:
            last_error = exc
            print(f"Failed from this URL: {exc}")

    raise RuntimeError(f"Could not download {file_path.name} from all URLs") from last_error


def read_image_bgr(image_path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def align_face_on_original(image_bgr, landmarks, output_size=FACE_ALIGNMENT_SIZE):
    dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    src = landmarks.astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None

    return cv2.warpAffine(
        image_bgr,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def load_arcface_model():
    ensure_file(ARCFACE_MODEL_PATH, ARCFACE_MODEL_URLS, min_size_bytes=5_000_000)

    providers = ["CPUExecutionProvider"]
    if DEVICE.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(ARCFACE_MODEL_PATH), providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def extract_embedding_arcface(face_aligned_bgr, arcface_session, input_name, output_name):
    face_rgb = cv2.cvtColor(face_aligned_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    face_rgb = face_rgb.astype(np.float32) / 255.0
    face_rgb = (face_rgb - 0.5) / 0.5

    face_input = np.transpose(face_rgb, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    embedding = arcface_session.run([output_name], {input_name: face_input})[0][0]
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding.astype(np.float32)


def load_yunet_detector():
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
            return detector, "CUDA"
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


def detect_largest_face_with_landmarks(image_bgr, face_detector):
    height, width = image_bgr.shape[:2]
    face_detector.setInputSize((width, height))
    _, detections = face_detector.detect(image_bgr)
    if detections is None or len(detections) == 0:
        return None, None

    best = max(detections, key=lambda det: det[2] * det[3])
    x, y, w, h = best[:4]
    bbox = (int(x), int(y), int(w), int(h))

    landmarks = np.array(
        [
            [best[4], best[5]],
            [best[6], best[7]],
            [best[8], best[9]],
            [best[10], best[11]],
            [best[12], best[13]],
        ],
        dtype=np.float32,
    )

    return bbox, landmarks


def extract_face_feature_from_bgr(
    image_bgr,
    face_detector,
    arcface_session,
    arcface_input_name,
    arcface_output_name,
):
    bbox, landmarks = detect_largest_face_with_landmarks(image_bgr, face_detector)
    if bbox is None or landmarks is None:
        return None

    aligned_bgr = align_face_on_original(image_bgr, landmarks, output_size=FACE_ALIGNMENT_SIZE)
    if aligned_bgr is None or aligned_bgr.size == 0:
        return None

    return extract_embedding_arcface(
        aligned_bgr,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
    )


def find_identity_dirs(dataset_root):
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


def build_or_load_gallery_features(
    dataset_root,
    cache_file,
    face_detector,
    arcface_session,
    arcface_input_name,
    arcface_output_name,
    max_images_per_person=GALLERY_IMAGES_PER_PERSON,
):
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
        raise RuntimeError("No identity folders with images found.")

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

            feat = extract_face_feature_from_bgr(
                img_bgr,
                face_detector,
                arcface_session,
                arcface_input_name,
                arcface_output_name,
            )
            if feat is None:
                continue

            features.append(feat)
            image_paths.append(str(img_path))
            labels.append(str(person_dir.relative_to(dataset_root)))

    if not features:
        raise RuntimeError("No valid face features found.")

    features = np.asarray(features, dtype=np.float32)
    np.savez_compressed(
        cache_file,
        features=features,
        image_paths=np.array(image_paths, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"Built and cached: {len(features)} samples")
    return features, image_paths, labels


def capture_single_frame_from_webcam(face_detector, detector_mode):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    captured = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        preview = frame.copy()
        bbox, landmarks = detect_largest_face_with_landmarks(preview, face_detector)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (lx, ly) in landmarks.astype(int):
                cv2.circle(preview, (lx, ly), 2, (0, 255, 255), -1)

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
    query_norm = np.linalg.norm(query) + 1e-8
    gallery_norms = np.linalg.norm(gallery, axis=1) + 1e-8
    return (gallery @ query) / (gallery_norms * query_norm)


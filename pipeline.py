from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from PIL import Image
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

PIPELINE_NOT_IMPLEMENTED = "Pipeline (chua co)"
PIPELINE_DL_MODELS = "DL Models (ResNet18 + YuNet)"
LABEL_TOP_K = 3
ACCEPT_THRESHOLD = 0.42


def ensure_file(file_path, url, min_size_bytes=10000):
	if file_path.exists() and file_path.stat().st_size >= min_size_bytes:
		return file_path

	if file_path.exists():
		file_path.unlink()

	urlretrieve(url, str(file_path))

	if file_path.stat().st_size < min_size_bytes:
		raise RuntimeError(f"Downloaded file looks invalid: {file_path}")

	return file_path


def read_image_bgr(image_path):
	data = np.fromfile(str(image_path), dtype=np.uint8)
	if data.size == 0:
		return None
	return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_embedding_model():
	weights = ResNet18_Weights.DEFAULT
	model = resnet18(weights=weights)
	model.fc = torch.nn.Identity()
	model = model.to(DEVICE)
	model.eval()
	return model, weights.transforms()


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


def detect_largest_face(face_detector, image_bgr):
	height, width = image_bgr.shape[:2]
	face_detector.setInputSize((width, height))
	_, detections = face_detector.detect(image_bgr)
	if detections is None or len(detections) == 0:
		return None

	best = max(detections, key=lambda det: det[2] * det[3])
	x, y, w, h = best[:4]
	return int(x), int(y), int(w), int(h)


def expand_face_box(box, image_shape, scale=0.2):
	x, y, w, h = box
	pad_w = int(w * scale)
	pad_h = int(h * scale)
	h_img, w_img = image_shape[:2]

	x1 = max(0, x - pad_w)
	y1 = max(0, y - pad_h)
	x2 = min(w_img, x + w + pad_w)
	y2 = min(h_img, y + h + pad_h)
	return x1, y1, x2 - x1, y2 - y1


def _embedding_from_face_bgr(embedding_model, embedding_preprocess, face_bgr):
	face_bgr = cv2.resize(face_bgr, FACE_SIZE, interpolation=cv2.INTER_AREA)
	face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
	face_rgb = cv2.resize(face_rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
	face_image = Image.fromarray(face_rgb)
	input_tensor = embedding_preprocess(face_image).unsqueeze(0).to(DEVICE)

	with torch.inference_mode():
		embedding = embedding_model(input_tensor).squeeze(0).cpu().numpy().astype(np.float32)

	embedding /= np.linalg.norm(embedding) + 1e-8
	return embedding


def _extract_face_crop(face_detector, image_bgr):
	box = detect_largest_face(face_detector, image_bgr)
	if box is None:
		return None

	x, y, w, h = expand_face_box(box, image_bgr.shape, scale=0.2)
	face_bgr = image_bgr[y:y + h, x:x + w]
	if face_bgr.size == 0:
		return None
	return face_bgr


def extract_face_feature_from_bgr(face_detector, embedding_model, embedding_preprocess, image_bgr):
	face_bgr = _extract_face_crop(face_detector, image_bgr)
	if face_bgr is None:
		return None
	return _embedding_from_face_bgr(embedding_model, embedding_preprocess, face_bgr)


def extract_stable_face_feature_from_bgr(face_detector, embedding_model, embedding_preprocess, image_bgr):
	face_bgr = _extract_face_crop(face_detector, image_bgr)
	if face_bgr is None:
		return None

	variants = [face_bgr, cv2.flip(face_bgr, 1)]
	ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
	ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
	variants.append(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))

	embeddings = [
		_embedding_from_face_bgr(embedding_model, embedding_preprocess, v)
		for v in variants
	]
	mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
	mean_emb /= np.linalg.norm(mean_emb) + 1e-8
	return mean_emb


def aggregate_label_scores(scores, labels, top_k=3):
	label_to_scores = {}
	for score, label in zip(scores, labels):
		label_to_scores.setdefault(label, []).append(float(score))

	label_scores = {}
	for label, vals in label_to_scores.items():
		sorted_vals = sorted(vals, reverse=True)
		use_vals = sorted_vals[:max(1, top_k)]
		label_scores[label] = float(np.mean(use_vals))
	return label_scores


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


def build_or_load_gallery_features(face_detector, embedding_model, embedding_preprocess, dataset_root, cache_file, max_images_per_person=GALLERY_IMAGES_PER_PERSON):
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

	if cache_file.exists():
		data = np.load(cache_file, allow_pickle=True)
		features = data["features"]
		image_paths = data["image_paths"].tolist()
		labels = data["labels"].tolist()
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

			feat = extract_face_feature_from_bgr(
				face_detector,
				embedding_model,
				embedding_preprocess,
				img_bgr,
			)
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
	return features, image_paths, labels


def cosine_similarity_matrix(query, gallery):
	query_norm = np.linalg.norm(query) + 1e-8
	gallery_norms = np.linalg.norm(gallery, axis=1) + 1e-8
	return (gallery @ query) / (gallery_norms * query_norm)


class EmptyPipeline:
	name = PIPELINE_NOT_IMPLEMENTED

	def run(self, _img_rgb):
		pass


class DLPipeline:
	name = PIPELINE_DL_MODELS

	def __init__(self):
		self._ready = False
		self.embedding_model = None
		self.embedding_preprocess = None
		self.face_detector = None
		self.detector_mode = "CPU"
		self.gallery_features = None
		self.gallery_paths = None
		self.gallery_labels = None

	def _ensure_ready(self):
		if self._ready:
			return

		self.embedding_model, self.embedding_preprocess = load_embedding_model()
		self.face_detector, self.detector_mode = load_yunet_detector()
		self.gallery_features, self.gallery_paths, self.gallery_labels = build_or_load_gallery_features(
			self.face_detector,
			self.embedding_model,
			self.embedding_preprocess,
			DATASET_ROOT,
			CACHE_FILE,
			max_images_per_person=GALLERY_IMAGES_PER_PERSON,
		)
		self._ready = True

	def run(self, img_rgb):
		if img_rgb is None:
			return "No image captured from camera!", None, []

		self._ensure_ready()

		logs = []
		img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

		logs.append(f"Embedding model ready on {DEVICE.type}")
		logs.append(f"Face detector mode: {self.detector_mode}")
		logs.append(f"Gallery size: {len(self.gallery_features)}")

		query_feature = extract_stable_face_feature_from_bgr(
			self.face_detector,
			self.embedding_model,
			self.embedding_preprocess,
			img_bgr,
		)
		if query_feature is None:
			return "No face detected in captured image.", None, []

		scores = cosine_similarity_matrix(query_feature, self.gallery_features)
		best_idx = int(np.argmax(scores))
		best_path = self.gallery_paths[best_idx]

		label_scores = aggregate_label_scores(scores, self.gallery_labels, top_k=LABEL_TOP_K)
		best_label, best_score = max(label_scores.items(), key=lambda kv: kv[1])

		best_bgr = read_image_bgr(best_path)
		if best_bgr is None:
			raise RuntimeError("Failed to load best-match image.")

		best_rgb = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB)

		logs.append(f"Best match folder: {best_label}")
		logs.append(f"Best match image: {best_path}")
		logs.append(f"Label score(top-{LABEL_TOP_K} mean): {best_score:.4f}")

		if best_score < ACCEPT_THRESHOLD:
			logs.append(f"Decision: Unknown (score < {ACCEPT_THRESHOLD:.2f})")
			return "\n".join(logs), None, [img_rgb]

		diagrams = [img_rgb, best_rgb]
		return "\n".join(logs), best_rgb, diagrams


_PIPELINES = {
	PIPELINE_NOT_IMPLEMENTED: EmptyPipeline(),
	PIPELINE_DL_MODELS: DLPipeline(),
}


def list_pipeline_names():
	return list(_PIPELINES.keys())


def run_selected_pipeline(pipeline_name, img_rgb):
	selected = _PIPELINES.get(pipeline_name)
	if selected is None:
		return f"Unknown pipeline: {pipeline_name}", None, []

	if isinstance(selected, EmptyPipeline):
		return "Pipeline nay chua duoc cai dat (pass).", None, []

	return selected.run(img_rgb)

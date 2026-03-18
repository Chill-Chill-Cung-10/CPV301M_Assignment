"""
Pipeline module for ArcFace face recognition.
Uses 6-step pipeline: YuNet → landmarks → align → RGB → 112×112 → ArcFace → L2 normalize
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from arcface_core import (
    DEVICE,
    DATASET_ROOT,
    CACHE_FILE,
    GALLERY_IMAGES_PER_PERSON,
    load_arcface_model,
    load_yunet_detector,
    extract_face_feature_from_bgr,
    build_or_load_gallery_features,
    capture_single_frame_from_webcam,
    read_image_bgr,
    cosine_similarity_matrix,
)


PIPELINE_ARCFACE = "ArcFace (YuNet + Landmarks + Alignment)"
PIPELINE_DEMO = "Demo (Load cache + Webcam + Match + Display)"


def get_pipeline_names():
    """List available pipelines."""
    return [PIPELINE_ARCFACE, PIPELINE_DEMO]


def run_arcface_pipeline():
    """Full ArcFace face recognition pipeline."""
    # Load models
    arcface_session, arcface_input_name, arcface_output_name = load_arcface_model()
    face_detector, detector_mode = load_yunet_detector()

    print(f"Device: {DEVICE}")
    print(f"Face detector: {detector_mode}")
    
    # Step 1: Load or build gallery
    print("\nStep 1: Building/loading gallery features...")
    gallery_features, gallery_paths, gallery_labels = build_or_load_gallery_features(
        DATASET_ROOT,
        CACHE_FILE,
        face_detector,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
        max_images_per_person=GALLERY_IMAGES_PER_PERSON,
    )
    print(f"Gallery loaded: {len(gallery_features)} samples")

    # Step 2: Capture webcam
    print("\nStep 2: Capturing from webcam...")
    query_bgr = capture_single_frame_from_webcam(face_detector, detector_mode)
    if query_bgr is None:
        print("No image captured.")
        return

    # Step 3: Extract query feature
    print("\nStep 3: Extracting query feature...")
    query_feature = extract_face_feature_from_bgr(
        query_bgr,
        face_detector,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
    )
    if query_feature is None:
        print("No face detected in query image.")
        return

    # Step 4: Find best match
    print("\nStep 4: Finding best match...")
    scores = cosine_similarity_matrix(query_feature, gallery_features)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_path = gallery_paths[best_idx]
    best_label = gallery_labels[best_idx]

    print(f"Best match: {best_label}")
    print(f"Best match image: {best_path}")
    print(f"Similarity score: {best_score:.4f}")

    # Step 5: Display results
    print("\nStep 5: Displaying results...")
    best_bgr = read_image_bgr(best_path)
    if best_bgr is None:
        print("Failed to load best-match image.")
        return

    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    best_rgb = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(query_rgb)
    plt.title("Captured Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(best_rgb)
    plt.title(f"Best Match (ID={best_label})\nScore={best_score:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Done!")


def run_demo_pipeline():
    """Demo: Load cache, capture, find match, display."""
    # Load models
    arcface_session, arcface_input_name, arcface_output_name = load_arcface_model()
    face_detector, detector_mode = load_yunet_detector()

    # Load gallery
    gallery_features, gallery_paths, gallery_labels = build_or_load_gallery_features(
        DATASET_ROOT,
        CACHE_FILE,
        face_detector,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
        max_images_per_person=GALLERY_IMAGES_PER_PERSON,
    )

    # Capture
    query_bgr = capture_single_frame_from_webcam(face_detector, detector_mode)
    if query_bgr is None:
        return

    # Extract
    query_feature = extract_face_feature_from_bgr(
        query_bgr,
        face_detector,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
    )
    if query_feature is None:
        return

    # Find match
    scores = cosine_similarity_matrix(query_feature, gallery_features)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_path = gallery_paths[best_idx]
    best_label = gallery_labels[best_idx]

    # Display
    best_bgr = read_image_bgr(best_path)
    if best_bgr is None:
        return

    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    best_rgb = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(query_rgb)
    plt.title("Captured")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(best_rgb)
    plt.title(f"Match: {best_label} (Score: {best_score:.4f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_selected_pipeline(pipeline_name):
    """Run selected pipeline by name."""
    if pipeline_name == PIPELINE_ARCFACE:
        return run_arcface_pipeline()
    elif pipeline_name == PIPELINE_DEMO:
        return run_demo_pipeline()
    else:
        print(f"Unknown pipeline: {pipeline_name}")

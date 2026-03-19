"""
Gradio UI for ArcFace Face Recognition
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path

from arcface_core import (
    DEVICE,
    DATASET_ROOT,
    CACHE_FILE,
    GALLERY_IMAGES_PER_PERSON,
    load_arcface_model,
    load_yunet_detector,
    extract_face_feature_from_bgr,
    build_or_load_gallery_features,
    read_image_bgr,
    cosine_similarity_matrix,
)


# Global models (load once)
try:
    arcface_session, arcface_input_name, arcface_output_name = load_arcface_model()
    face_detector, detector_mode = load_yunet_detector()
    
    gallery_features, gallery_paths, gallery_labels = build_or_load_gallery_features(
        DATASET_ROOT,
        CACHE_FILE,
        face_detector,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
        max_images_per_person=GALLERY_IMAGES_PER_PERSON,
    )
    print(f"✓ Models loaded. Gallery: {len(gallery_features)} samples")
except Exception as e:
    print(f"Error loading models: {e}")
    gallery_features = None


def process_face_image(image_bgr):
    """Process uploaded face image and find best match."""
    if image_bgr is None:
        return None, "No image uploaded"

    if gallery_features is None:
        return None, "Gallery not loaded"

    # Extract feature
    query_feature = extract_face_feature_from_bgr(
        image_bgr,
        face_detector,
        arcface_session,
        arcface_input_name,
        arcface_output_name,
    )
    if query_feature is None:
        return None, "No face detected"

    # Find match
    scores = cosine_similarity_matrix(query_feature, gallery_features)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_path = gallery_paths[best_idx]
    best_label = gallery_labels[best_idx]

    # Load best match image
    best_bgr = read_image_bgr(best_path)
    if best_bgr is None:
        return None, f"Error loading best match image"

    best_rgb = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB)

    result_text = f"""
**Best Match Found**
- Identity: {best_label}
- Similarity Score: {best_score:.4f}
- Image: {Path(best_path).name}
- Device: {DEVICE}
"""

    return best_rgb, result_text


def create_ui():
    """Create Gradio interface."""
    with gr.Blocks(title="ArcFace Face Recognition") as demo:
        gr.Markdown("# ArcFace Face Recognition")
        gr.Markdown("Upload a face image to find matches in gallery")

        with gr.Row():
            # Input area
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Face Image",
                    type="numpy",
                    height=400,
                )
                process_btn = gr.Button("🔍 Find Match", size="lg")

            # Output area
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Best Match",
                    type="numpy",
                    height=400,
                )
                result_text = gr.Markdown(label="Results")

        # Process button click
        process_btn.click(
            fn=process_face_image,
            inputs=[input_image],
            outputs=[output_image, result_text],
        )

        gr.Markdown(f"**Status**: Gallery loaded with {len(gallery_features) if gallery_features is not None else 0} samples")

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(show_error=True)


import gradio as gr
import numpy as np
import cv2
import time

def process_pipeline(img):
    logs = ""
    diagrams = []

    if img is None:
        return "No image captured from camera!", None, []

    logs += "Step 1: Convert to grayscale...\n"

    # Gradio trả ảnh dạng RGB
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    diagrams.append(gray)
    time.sleep(0.5)

    logs += "Step 2: Detect edges...\n"
    edges = cv2.Canny(gray, 100, 200)
    diagrams.append(edges)
    time.sleep(0.5)

    logs += "Step 3: Dilate edges...\n"
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel)
    diagrams.append(dilated)

    return logs, dilated, diagrams


with gr.Blocks() as demo:

    with gr.Row():   # chia ngang

        # ===== INPUT AREA =====
        with gr.Column(scale=1):

            camera = gr.Image(
                sources="webcam",
                type="numpy",
                label="Camera"
            )

            capture_btn = gr.Button("📷 Take Photo")

            console = gr.Textbox(
                label="Processing Pipeline",
                lines=12
            )

        # ===== OUTPUT AREA =====
        with gr.Column(scale=1):

            result_img = gr.Image(label="Processed Image")

            pipeline_gallery = gr.Gallery(
                label="Pipeline Steps",
                columns=3
            )

    capture_btn.click(
        fn=process_pipeline,
        inputs=camera,
        outputs=[console, result_img, pipeline_gallery]
    )

demo.launch()
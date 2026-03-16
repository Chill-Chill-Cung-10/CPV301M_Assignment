import gradio as gr
from pipeline import list_pipeline_names, run_selected_pipeline

def process_pipeline(pipeline_name, img):
    return run_selected_pipeline(pipeline_name, img)


with gr.Blocks() as demo:

    with gr.Row():   # chia ngang

        # ===== INPUT AREA =====
        with gr.Column(scale=1):
            pipeline_selector = gr.Dropdown(
                choices=list_pipeline_names(),
                value=list_pipeline_names()[1],
                label="Select Pipeline",
            )

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
        inputs=[pipeline_selector, camera],
        outputs=[console, result_img, pipeline_gallery]
    )

demo.launch()
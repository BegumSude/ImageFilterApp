import cv2 as cv
import numpy as np
import gradio as gr

def apply_gaussian_blur(frame, density):
    ksize = int(density) * 2 + 1
    return cv.GaussianBlur(frame, (ksize, ksize), 0)

def apply_sharpening_filter(frame, density):
    kernel = np.array([[-1, -1, -1], [-1, 8 * density, -1], [-1, -1, -1]])
    return cv.filter2D(frame, -1, kernel)

def apply_edge_detection(frame, density):
    return cv.Canny(frame, 100, 100 * density)

def apply_invert_filter(frame, density):
    return cv.bitwise_not(frame)

def adjust_brightness_contrast(frame, density):
    alpha = 1.0 + density / 50.0
    beta = density * 2
    return cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame, density):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def apply_sepia_filter(frame, density):
    sepia_filter = np.array([[0.272 * density, 0.534 * density, 0.131 * density],
                             [0.349 * density, 0.686 * density, 0.168 * density],
                             [0.393 * density, 0.769 * density, 0.189 * density]])
    return cv.transform(frame, sepia_filter)

def apply_sketch_filter(frame, density):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inv = cv.bitwise_not(gray)
    blur = cv.GaussianBlur(inv, (21, 21), 0)
    sketch = cv.divide(gray, 255 - blur, scale=256)
    return cv.multiply(sketch, density)

def apply_cartoon_filter(frame, density):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 7)
    edges = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 10)
    color = cv.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv.bitwise_and(color, color, mask=edges)
    return cv.multiply(cartoon, density)

def apply_pixelate_filter(frame, density):
    pixel_size = int(density) + 1
    h, w = frame.shape[:2]
    temp = cv.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv.INTER_LINEAR)
    return cv.resize(temp, (w, h), interpolation=cv.INTER_NEAREST)

def apply_emboss_filter(frame, density):
    kernel = np.array([[-2, -1, 0], [-1, 1 * density, 1], [0, 1, 2]])
    return cv.filter2D(frame, -1, kernel)

def apply_brightness_contrast_adjustment(frame, density):
    return adjust_brightness_contrast(frame, density)

def apply_sepia_tone(frame, density):
    return apply_sepia_filter(frame, density)


def apply_filter(input_image, filter_type, density):

    frame = np.array(input_image)

    if filter_type == "Gaussian Blur":
        result = apply_gaussian_blur(frame, density)
    elif filter_type == "Sharpen":
        result = apply_sharpening_filter(frame, density)
    elif filter_type == "Edge Detection":
        result = apply_edge_detection(frame, density)
    elif filter_type == "Invert":
        result = apply_invert_filter(frame, density)
    elif filter_type == "Brightness":
        result = adjust_brightness_contrast(frame, density)
    elif filter_type == "GrayScale":
        result = apply_grayscale_filter(frame, density)
    elif filter_type == "Sepia":
        result = apply_sepia_filter(frame, density)
    elif filter_type == "Sketch":
        result = apply_sketch_filter(frame, density)
    elif filter_type == "Cartoon":
        result = apply_cartoon_filter(frame, density)
    elif filter_type == "Pixelate":
        result = apply_pixelate_filter(frame, density)
    elif filter_type == "Emboss":
        result = apply_emboss_filter(frame, density)
    elif filter_type == "Brightness/Contrast":
        result = apply_brightness_contrast_adjustment(frame, density)
    elif filter_type == "Sepia Tone":
        result = apply_sepia_tone(frame, density)
    else:
        result = frame

    return result

with gr.Blocks(css="""
    #filter-dropdown {
        width: 300px;
        margin: 0 auto;
    }
    #apply-button {
        background-color: #8B4513;
        color: white;
        font-weight: bold;
        margin-top: 20px;
    }
    #apply-button:hover {
        background-color: #8B4513;
    }
    #input-image, #output-image {
        width: 100%;
        border-radius: 10px;
    }
    h1 {
        text-align: center;
        color: #8B4513;
    }
    p {
        text-align: center;
        font-size: 20px;
    }
""") as demo:

    gr.Markdown("""
        <h1>🖼️ Image Filter Application  🖼️</h1>
        <p>Select a filter and apply it to your image :) Enjoy!</p>
    """)

    with gr.Row():
        with gr.Column():

            filter_type = gr.Radio(
                label="Choose a filter:",
                choices=["Gaussian Blur", "Sharpen", "Edge Detection", "Invert", "Brightness", "GrayScale", "Sepia", "Sketch", "Cartoon", "Pixelate", "Emboss", "Brightness/Contrast", "Sepia Tone"],
                value="Gaussian Blur",
                elem_id="filter-radio"
            )

            density_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=0.1,
                label="Filter Intensity (Density)",
                value=3,
                elem_id="density-slider"
            )

            input_image = gr.Image(label="Upload Image", elem_id="input-image")

            apply_button = gr.Button("Apply Filter", elem_id="apply-button")

        with gr.Column():
            output_image = gr.Image(label="Filtered Image", elem_id="output-image")

    apply_button.click(fn=apply_filter, inputs=[input_image, filter_type, density_slider], outputs=output_image)

demo.launch(share=True)

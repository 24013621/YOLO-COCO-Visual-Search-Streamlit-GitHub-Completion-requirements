**Experiment No:** 04
**Date:**

# Real-Time Object Detection using YOLOv8 (COCO) with Streamlit

## AIM

To build and deploy a real-time object detection web application using a YOLOv8 model pretrained on the COCO dataset, run locally through a Conda environment in Visual Studio Code, and served with Streamlit.

## ABSTRACT / INTRODUCTION

Object detection is a core computer vision task that involves both localizing objects within an image (drawing bounding boxes) and classifying them into predefined categories. This project uses **YOLOv8** (You Only Look Once, version 8), a fast single-stage object detector from Ultralytics, pretrained on the **COCO (Common Objects in Context)** dataset covering 80 everyday object classes (person, car, dog, chair, laptop, etc.). A **Streamlit** web interface lets a user upload any image, runs YOLOv8 inference on it, and displays the annotated result (bounding boxes + class labels + confidence scores) alongside a table of all detected objects. The entire application is designed to be set up and run locally via a Conda environment inside VS Code, with either CPU-only or GPU-accelerated PyTorch installations.

## THEORY

### YOLO (You Only Look Once)

Unlike two-stage detectors (e.g. Faster R-CNN) that first propose regions and then classify them, YOLO frames detection as a single regression problem: a single neural network pass predicts bounding boxes and class probabilities directly from the full image, making it fast enough for real-time use. YOLOv8 improves on earlier versions with an anchor-free detection head, a more efficient backbone, and better training recipes, while remaining a drop-in "load weights and predict" model via the `ultralytics` Python package.

### Dataset & YOLO Model Details (COCO)

- **Dataset:** COCO (Common Objects in Context) — a large-scale dataset of ~330K images across **80 object categories** (person, bicycle, car, dog, cat, bottle, chair, laptop, etc.), widely used as the standard benchmark for object detection.
- **Model used:** `yolov8n.pt` (YOLOv8 "nano") — the smallest, fastest Ultralytics YOLOv8 checkpoint, already pretrained on COCO. It is downloaded automatically the first time `app.py` runs.
- **Why COCO pretraining matters:** the model can detect and label 80 common real-world object classes out-of-the-box, with no additional training required for this workshop.

## Architecture / Pipeline

1. User uploads an image through the Streamlit UI.
2. The image is converted from a PIL Image (RGB) to a NumPy array.
3. `YOLOv8n.predict(image, conf=threshold)` runs inference on the array.
4. The model returns bounding boxes, class labels, and confidence scores.
5. `result.plot()` draws the boxes/labels onto the image to produce the annotated output, and a detections table is built from the same results.
6. Both the annotated image and the detections table are displayed back in the Streamlit UI.

## ALGORITHM

### STEP 1: Environment Setup

Create and activate a Conda environment, then install PyTorch (CPU or GPU build depending on your system) plus the remaining dependencies from `requirements.txt`. See [Environment Setup](#environment-setup) below and `instruction.txt`.

### STEP 2: Load the Pretrained YOLOv8 Model

Load the `yolov8n.pt` checkpoint (pretrained on COCO) using the `ultralytics` package, cached with `st.cache_resource` so it loads only once per session.

### STEP 3: Build the Streamlit Interface

Create an image upload widget, a confidence-threshold slider, and side-by-side display panels for the original and annotated images.

### STEP 4: Run Inference

On each uploaded image, run `model.predict()`, draw the resulting bounding boxes/labels onto the image, and build a summary table of detected classes, confidences, and box coordinates.

### STEP 5: Display Results

Render the annotated image, detection table, and inference time back to the user in the browser.

### STEP 6: Deploy Locally

Run the app with `streamlit run app.py` from the activated Conda environment inside the VS Code terminal, and verify it opens correctly at `http://localhost:8501`.

## PROGRAM

See [`app.py`](./app.py) for the full, final application code. Core logic:

```python
from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image

MODEL_PATH = "yolov8n.pt"

@st.cache_resource(show_spinner="Loading YOLOv8 model (COCO weights)...")
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    return YOLO(model_path)

def run_detection(model, image: Image.Image, conf_threshold: float):
    results = model.predict(source=np.array(image), conf=conf_threshold, verbose=False)
    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return annotated_rgb, result

model = load_model()
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    annotated_rgb, result = run_detection(model, image, conf_threshold=0.25)
    st.image(annotated_rgb, caption="Detection Result")
```

## ENVIRONMENT SETUP

- Python 3.10 (via Conda)
- Visual Studio Code with the Python extension
- Full step-by-step commands are provided in [`instruction.txt`](./instruction.txt)

### GPU Installation Steps

```bash
conda create -n yolo-streamlit python=3.10 -y
conda activate yolo-streamlit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> Check your NVIDIA driver's supported CUDA version at
> https://pytorch.org/get-started/locally/ and adjust the `cu121` tag if needed.

### CPU Installation Steps

```bash
conda create -n yolo-streamlit python=3.10 -y
conda activate yolo-streamlit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## How to Run in VS Code using Conda

1. Open this project folder in VS Code (`File → Open Folder...`).
2. Open a new integrated terminal (`Terminal → New Terminal`).
3. Activate the environment: `conda activate yolo-streamlit`
4. Set the VS Code Python interpreter to the same environment:
   `Ctrl+Shift+P → Python: Select Interpreter → yolo-streamlit`
5. Confirm CUDA/CPU status if needed:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

## How to Deploy using Streamlit

From the activated Conda environment, in the VS Code terminal:

```bash
streamlit run app.py
```

This starts a local web server and opens the app in your browser at
`http://localhost:8501`. The first run downloads `yolov8n.pt` automatically
(requires internet access once).

## Output Screenshots

All screenshots are stored in the [`Screenshots/`](./Screenshots) folder. Required set:

| Screenshot | File |
|---|---|
| Conda environment activation in VS Code terminal | `Screenshots/01_conda_activation.png` |
| Running `streamlit run app.py` in terminal | `Screenshots/02_streamlit_run_terminal.png` |
| Streamlit web UI in browser | `Screenshots/03_streamlit_ui.png` |
| Object detection result screen | `Screenshots/04_detection_result.png` |

> ⚠️ Take these screenshots yourself while running the app on your own
> machine — do not copy or reuse images from elsewhere, per the
> academic integrity requirement.

## Enhancements / Innovations Added

- **Adjustable confidence threshold** slider so the user can tune sensitivity live, without editing code.
- **Cached model loading** (`st.cache_resource`) so the YOLOv8 model is loaded once per session, not on every image upload.
- **Side-by-side comparison** of the original vs. annotated image for easier visual inspection.
- **Structured detections table** (class, confidence, bounding box coordinates) alongside the visual output, useful for downstream analysis.
- **Inference timing** displayed to the user, useful for comparing CPU vs. GPU performance.

## Results & Conclusion

The YOLOv8n model, pretrained on the COCO dataset, successfully detects and localizes common real-world objects in user-uploaded images through a simple, locally-deployed Streamlit interface. The application runs entirely offline after the one-time weight download, works on both CPU-only and GPU-accelerated machines, and demonstrates a complete end-to-end computer-vision deployment pipeline: environment setup → model loading → inference → visualization → local web deployment.

---

## Repository Structure

```
.
├── app.py                # Final Streamlit + YOLOv8 application code
├── requirements.txt      # Python dependencies
├── instruction.txt       # Conda environment setup commands (CPU & GPU)
├── README.md
└── Screenshots/          # Required screenshots (see checklist above)
    ├── 01_conda_activation.png
    ├── 02_streamlit_run_terminal.png
    ├── 03_streamlit_ui.png
    └── 04_detection_result.png
```

## Submission Checklist

- [ ] `app.py` — final code
- [ ] `requirements.txt`
- [ ] All supporting files/folders
- [ ] `Screenshots/` folder with all 4 required screenshots (your own, not downloaded)
- [ ] `README.md` with all required sections (this file)
- [ ] Repository set to **Public** on GitHub
- [ ] Only the GitHub repository link submitted on Moodle (no ZIP/Drive links/video-only)

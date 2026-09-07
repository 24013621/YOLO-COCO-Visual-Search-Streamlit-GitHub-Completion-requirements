"""
Object Detection App using YOLOv8 (pretrained on COCO) + Streamlit
--------------------------------------------------------------------
Run with:
    streamlit run app.py
"""

import io
import time

import numpy as np
import streamlit as st
from PIL import Image

from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Object Detection (COCO)",
    page_icon="🎯",
    layout="wide",
)

MODEL_PATH = "yolov8n.pt"  # small/fast COCO-pretrained checkpoint, auto-downloaded on first run


@st.cache_resource(show_spinner="Loading YOLOv8 model (COCO weights)...")
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Load and cache the YOLOv8 model so it isn't reloaded on every interaction."""
    return YOLO(model_path)


def run_detection(model: YOLO, image: Image.Image, conf_threshold: float):
    """Run YOLO inference on a PIL image and return the annotated image + results."""
    results = model.predict(source=np.array(image), conf=conf_threshold, verbose=False)
    result = results[0]
    annotated_bgr = result.plot()  # numpy array (BGR) with boxes/labels drawn
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return annotated_rgb, result


def summarize_detections(result) -> "list[dict]":
    """Turn YOLO result boxes into a list of dicts for a clean Streamlit table."""
    rows = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            rows.append(
                {
                    "Class": names[cls_id],
                    "Confidence": round(float(box.conf.item()), 3),
                    "Box (x1, y1, x2, y2)": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )
    return rows


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
st.sidebar.markdown(
    """
**Model:** YOLOv8n (nano) — pretrained on the
[COCO dataset](https://cocodataset.org/) (80 object classes).

Adjust the confidence threshold to filter out low-confidence detections.
"""
)

# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.title("🎯 Real-Time Object Detection with YOLOv8 (COCO)")
st.write(
    "Upload an image and the app will detect and label objects "
    "(people, vehicles, animals, household items, etc.) using a YOLOv8 model "
    "pretrained on the COCO dataset."
)

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with st.spinner("Running detection..."):
        start = time.time()
        annotated_rgb, result = run_detection(model, image, conf_threshold)
        elapsed = time.time() - start

    with col2:
        st.subheader("Detection Result")
        st.image(annotated_rgb, use_column_width=True)

    st.success(f"Inference completed in {elapsed:.2f} seconds.")

    detections = summarize_detections(result)
    st.subheader(f"Detected Objects ({len(detections)})")
    if detections:
        st.table(detections)
    else:
        st.info("No objects detected above the current confidence threshold.")
else:
    st.info("👆 Upload an image to run object detection.")

st.markdown("---")
st.caption(
    "Name: <YOUR NAME>  |  Register No: <YOUR REGISTER NUMBER>  |  "
    "Model: YOLOv8n (Ultralytics) trained on COCO"
)

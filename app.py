import os
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

APP_TITLE = "CocoNexa"
DEFAULT_MODEL_PATH = Path("best.pt")

MATURITY_CLASSES = ["Young", "Semi-mature", "Mature", "Over mature"]
CLASS_COLORS = {
    "Young": (37, 99, 235),          # blue
    "Semi-mature": (22, 163, 74),    # green
    "Mature": (217, 119, 6),         # amber
    "Over mature": (220, 38, 38),    # red
}
UNKNOWN_COLOR = (107, 114, 128)

st.set_page_config(page_title="CocoNexa | Coconut Maturity Analyzer",
                   page_icon="🥥", layout="wide")

st.markdown("""
<style>
  .stApp { background: #f7faf7; }
  .hero {
    padding: 1.35rem 1.5rem; border-radius: 18px;
    background: linear-gradient(120deg, #123d2b, #24734b);
    color: white; margin-bottom: 1.2rem;
  }
  .hero h1 { color: white; margin: 0; font-size: 2.25rem; }
  .hero p { color: #e1f3e7; margin: .4rem 0 0 0; }
  .metric-card {
    background: white; border: 1px solid #e0e9e2;
    padding: 1rem; border-radius: 14px;
  }
  .small-muted { color: #607067; font-size: .9rem; }
</style>
<div class="hero">
  <h1>🥥 CocoNexa</h1>
  <p>AI Coconut Maturity Analyzer · Detect, count, and classify visible coconuts</p>
</div>
""", unsafe_allow_html=True)

def normalize_class_name(name):
    """Map common trained-model labels to the four requested maturity groups."""
    cleaned = " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())
    compact = cleaned.replace(" ", "")
    if compact in {"young", "immature", "youngcoconut", "immaturecoconut"}:
        return "Young"
    if compact in {"semimature", "semimaturecoconut", "semi", "semiripe"}:
        return "Semi-mature"
    if compact in {"mature", "maturecoconut", "fullymature"}:
        return "Mature"
    if compact in {"overmature", "overmaturecoconut", "too mature"}:
        return "Over mature"
    return str(name)

@st.cache_resource(show_spinner="Loading the coconut detection model…")
def load_model(model_path, model_bytes=None):
    # model_bytes is used for a model uploaded through the browser.
    if model_bytes is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(model_bytes)
            temp_path = tmp.name
        try:
            model = YOLO(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return model
    return YOLO(model_path)

with st.sidebar:
    st.header("⚙️ Analysis settings")
    st.caption("Choose the trained YOLO model and tune detection thresholds.")
    uploaded_model = st.file_uploader("Upload model weights (.pt)", type=["pt"])
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_threshold = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
    st.divider()
    st.markdown("**Bounding-box legend**")
    for label, color in CLASS_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0">'
            f'<span style="width:13px;height:13px;border-radius:3px;background:rgb{color};display:inline-block"></span>'
            f'<span>{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("Predictions depend on the training data and image quality.")

model_path = os.getenv("COCONEXA_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model_available = Path(model_path).is_file() or uploaded_model is not None

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("1. Upload coconut image")
    uploaded_image = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image. The model can only count coconuts it detects in this image."
    )
    if uploaded_image:
        try:
            original_image = Image.open(uploaded_image).convert("RGB")
            st.image(original_image, caption="Original image", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please upload a valid JPG or PNG.")

with right:
    st.subheader("2. Detection results")
    if not model_available:
        st.warning("A trained YOLO model is required before analysis can run.")
        st.markdown("""
**Add your trained model to enable predictions**

- Put your trained weights file in the project folder and name it `best.pt`, then deploy again; or
- Upload the `.pt` weights using **Upload model weights** in the sidebar.

The model must be trained to detect coconuts and classify them into the four maturity classes.
""")
        st.info("This app does not include fabricated predictions or a placeholder model.")
    elif not uploaded_image:
        st.info("Upload a coconut image to see bounding boxes, total count, and maturity-wise counts.")
    else:
        if st.button("🔍 Analyze coconuts", type="primary", use_container_width=True):
            try:
                with st.spinner("Detecting and classifying coconuts…"):
                    if uploaded_model is not None:
                        model = load_model("uploaded-model.pt", uploaded_model.getvalue())
                    else:
                        model = load_model(model_path)
                    import cv2
                    # Ultralytics expects OpenCV-style BGR arrays for NumPy image input.
                    image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                    result = model.predict(
                        source=image_array,
                        conf=confidence,
                        iou=iou_threshold,
                        imgsz=image_size,
                        verbose=False
                    )[0]

                    annotated_bgr = result.orig_img.copy()
                    counts = {label: 0 for label in MATURITY_CLASSES}
                    unknown_counts = {}
                    detections = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            score = float(box.conf.item())
                            raw_name = result.names.get(class_id, str(class_id))
                            maturity = normalize_class_name(raw_name)
                            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                            x1, y1, x2, y2 = xyxy
                            color = CLASS_COLORS.get(maturity, UNKNOWN_COLOR)

                            # Draw an individual bounding box and label with OpenCV.
                            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
                            label_text = f"{maturity} {score:.0%}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale, thickness = 0.58, 2
                            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                            label_y = max(y1, th + 12)
                            cv2.rectangle(annotated_bgr, (x1, label_y - th - 10),
                                          (x1 + tw + 8, label_y + baseline - 5), color, -1)
                            cv2.putText(annotated_bgr, label_text, (x1 + 4, label_y - 5),
                                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                            if maturity in counts:
                                counts[maturity] += 1
                            else:
                                unknown_counts[maturity] = unknown_counts.get(maturity, 0) + 1
                            detections.append({
                                "Maturity": maturity,
                                "Confidence": f"{score:.1%}",
                                "Bounding box (x1, y1, x2, y2)": str(xyxy)
                            })

                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    total_count = len(detections)
                    st.session_state["analysis"] = {
                        "annotated": annotated_rgb,
                        "counts": counts,
                        "unknown_counts": unknown_counts,
                        "total": total_count,
                        "detections": detections,
                    }
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.caption("Check that the uploaded weights are a compatible Ultralytics YOLO model.")

        analysis = st.session_state.get("analysis")
        if analysis:
            st.image(analysis["annotated"], caption="Detected coconuts with maturity labels", use_container_width=True)
            st.success(f"Analysis complete · {analysis['total']} coconut detection(s)")
            metrics = st.columns(4)
            for column, label in zip(metrics, MATURITY_CLASSES):
                column.metric(label, analysis["counts"][label])
            if analysis["unknown_counts"]:
                st.warning(
                    "Some model classes do not match the four requested maturity labels: "
                    + ", ".join(f"{k}: {v}" for k, v in analysis["unknown_counts"].items())
                    + ". Rename/retrain the model classes or update the class mapping in app.py."
                )
            st.markdown("#### Detection details")
            if analysis["detections"]:
                st.dataframe(analysis["detections"], use_container_width=True, hide_index=True)
                from PIL import Image as PILImage
                import io
                buffer = io.BytesIO()
                PILImage.fromarray(analysis["annotated"]).save(buffer, format="PNG")
                st.download_button(
                    "⬇️ Download annotated image",
                    data=buffer.getvalue(),
                    file_name="coconexa_annotated.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("No coconuts were detected at this confidence threshold. Try lowering the threshold or using a clearer image.")

st.divider()
st.caption("CocoNexa is a decision-support prototype. Counts represent visible coconuts detected in the uploaded image; results are not guaranteed and should be validated against labeled test images.")

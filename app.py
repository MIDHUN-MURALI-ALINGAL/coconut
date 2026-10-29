import os
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS

APP_TITLE = "CocoNexa"
DEFAULT_MODEL_PATH = Path("best.pt")

MATURITY_CLASSES = ["Young", "Semi-mature", "Mature", "Over mature"]
CLASS_COLORS = {
    "Young": (37, 99, 235),        # blue
    "Semi-mature": (22, 163, 74),  # green
    "Mature": (217, 119, 6),       # amber
    "Over mature": (220, 38, 38),  # red
}
CLASS_HEX = {
    "Young": "#2563eb",
    "Semi-mature": "#16a34a",
    "Mature": "#d97706",
    "Over mature": "#dc2626",
}
UNKNOWN_COLOR = (107, 114, 128)

st.set_page_config(
    page_title="CocoNexa | Coconut Maturity Analyzer",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background: #f5f8f5; }
  [data-testid="stHeader"] { background: rgba(245,248,245,.9); }
  .hero {
    padding: 1.3rem 1.6rem; border-radius: 18px;
    background: linear-gradient(115deg, #123d2b, #24734b);
    color: white; margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(18,61,43,.12);
  }
  .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
  .hero p { color: #e1f3e7; margin: .45rem 0 0 0; }
  .section-card {
    background: white; border: 1px solid #e1e9e2;
    border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
  }
  .class-row {
    display:flex; align-items:center; justify-content:space-between;
    gap:10px; padding: 11px 12px; border:1px solid #e8eee9;
    border-radius: 11px; margin: 8px 0; background: #fff;
  }
  .class-name { display:flex; align-items:center; gap:9px; font-weight:600; }
  .dot { width:12px; height:12px; border-radius:4px; display:inline-block; }
  .class-count { font-size:1.35rem; font-weight:750; }
  .muted { color:#64746a; font-size:.88rem; }
  div[data-testid="stMetric"] {
    background:white; border:1px solid #e1e9e2;
    padding: 12px 14px; border-radius: 12px;
  }
  .stButton > button[kind="primary"] { background:#17613d; border-color:#17613d; }
</style>
<div class="hero">
  <h1>🥥 CocoNexa</h1>
  <p>AI Coconut Maturity Analyzer · Upload or capture an image to detect, count, and classify coconuts.</p>
</div>
""", unsafe_allow_html=True)


def normalize_class_name(name):
    """Map common trained-model labels to the four dashboard classes."""
    cleaned = " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())
    compact = cleaned.replace(" ", "")
    aliases = {
        "young": "Young",
        "immature": "Young",
        "youngcoconut": "Young",
        "immaturecoconut": "Young",
        "semi": "Semi-mature",
        "semimature": "Semi-mature",
        "semimaturecoconut": "Semi-mature",
        "semiripe": "Semi-mature",
        "mature": "Mature",
        "maturecoconut": "Mature",
        "fullymature": "Mature",
        "overmature": "Over mature",
        "overmaturecoconut": "Over mature",
        "toomature": "Over mature",
    }
    return aliases.get(compact, str(name))


@st.cache_resource(show_spinner="Loading the trained coconut model…")
def load_model_from_path(model_path):
    return YOLO(model_path)


@st.cache_resource(show_spinner="Loading uploaded model weights…")
def load_model_from_bytes(model_bytes):
    # Keep the temporary file available while YOLO loads it, then remove it.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(model_bytes)
            temp_path = tmp.name
        model = YOLO(temp_path)
        return model
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def make_voice_summary(counts, total, language):
    if language == "Malayalam":
        summary = (
            f"കൊക്കനക്സ വിശകലനം പൂർത്തിയായി. കണ്ടെത്തിയ ആകെ തേങ്ങകൾ {total}. "
            f"ഇളം തേങ്ങകൾ {counts['Young']}. "
            f"അർദ്ധ പക്വമായ തേങ്ങകൾ {counts['Semi-mature']}. "
            f"പക്വമായ തേങ്ങകൾ {counts['Mature']}. "
            f"അതിർപക്വമായ തേങ്ങകൾ {counts['Over mature']}."
        )
        lang_code = "ml"
    else:
        summary = (
            f"CocoNexa analysis complete. Total coconuts detected: {total}. "
            f"Young: {counts['Young']}. Semi-mature: {counts['Semi-mature']}. "
            f"Mature: {counts['Mature']}. Over mature: {counts['Over mature']}."
        )
        lang_code = "en"
    audio = io.BytesIO()
    gTTS(text=summary, lang=lang_code, slow=False).write_to_fp(audio)
    audio.seek(0)
    return audio.getvalue(), summary


def run_detection(model, image_pil, conf, iou, imgsz):
    # Convert RGB image to BGR before passing a NumPy image to Ultralytics.
    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(
        source=bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False
    )[0]

    counts = {label: 0 for label in MATURITY_CLASSES}
    unknown_counts = {}
    detections = []
    annotated = result.orig_img.copy()

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            raw_name = result.names.get(class_id, str(class_id))
            maturity = normalize_class_name(raw_name)
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            color = CLASS_COLORS.get(maturity, UNKNOWN_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label_text = f"{maturity} {score:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.58, 2
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            label_y = max(y1, th + 12)
            cv2.rectangle(
                annotated, (x1, label_y - th - 10),
                (x1 + tw + 8, label_y + baseline - 5), color, -1
            )
            cv2.putText(
                annotated, label_text, (x1 + 4, label_y - 5), font,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )

            if maturity in counts:
                counts[maturity] += 1
            else:
                unknown_counts[maturity] = unknown_counts.get(maturity, 0) + 1

            detections.append({
                "Maturity": maturity,
                "Confidence": f"{score:.1%}",
                "Bounding box (x1, y1, x2, y2)": str(xyxy),
            })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, counts, unknown_counts, detections


# Sidebar controls
with st.sidebar:
    st.header("⚙️ Analysis settings")
    st.caption("Choose model weights and adjust detection sensitivity.")
    uploaded_model = st.file_uploader("Upload trained YOLO model (.pt)", type=["pt"])
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_threshold = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.select_slider(
        "Inference image size", options=[320, 416, 512, 640, 768, 960], value=640
    )
    voice_language = st.selectbox("Voice assistance language", ["English", "Malayalam"])
    st.divider()
    st.markdown("**Bounding-box legend**")
    for label in MATURITY_CLASSES:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:8px 0">'
            f'<span style="width:13px;height:13px;border-radius:3px;'
            f'background:{CLASS_HEX[label]};display:inline-block"></span>'
            f'<span>{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("Predictions depend on your trained model and image quality.")

# Image source selector: file upload or live camera capture.
st.subheader("📷 Choose coconut image")
image_source = st.radio(
    "Image source", ["Upload image", "Take a picture"], horizontal=True,
    label_visibility="collapsed"
)
uploaded_image = None
if image_source == "Upload image":
    uploaded_image = st.file_uploader(
        "Upload a coconut image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="coconut_upload",
    )
else:
    uploaded_image = st.camera_input(
        "Take a picture of coconuts",
        help="Camera access depends on your browser/device permissions. On phones, use HTTPS."
    )

model_path = os.getenv("COCONEXA_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model_exists = Path(model_path).is_file()
model_ready = model_exists or uploaded_model is not None

# Main image and persistent right-hand dashboard.
image_col, dashboard_col = st.columns([1.75, 1], gap="large")

with image_col:
    st.markdown("### 🖼️ Image and bounding boxes")
    if uploaded_image is not None:
        try:
            original_image = Image.open(uploaded_image).convert("RGB")
            st.image(original_image, caption="Selected coconut image", use_container_width=True)
        except Exception:
            st.error("Could not open this image. Please try a JPG or PNG.")
            original_image = None
    else:
        original_image = None
        st.info("Upload an image or take a picture. Detected coconuts will be outlined with colour-coded boxes here.")

    if not model_ready:
        st.warning(
            "A trained YOLO `.pt` model is needed for real detections. Add `best.pt` "
            "to the repository root or upload the weights in the sidebar."
        )

    analyze_clicked = st.button(
        "🔍 Analyze coconuts", type="primary", use_container_width=True,
        disabled=(original_image is None or not model_ready),
    )

    if analyze_clicked and original_image is not None:
        try:
            with st.spinner("Detecting coconuts and estimating maturity…"):
                if uploaded_model is not None:
                    model = load_model_from_bytes(uploaded_model.getvalue())
                else:
                    model = load_model_from_path(model_path)
                annotated, counts, unknown_counts, detections = run_detection(
                    model, original_image, confidence, iou_threshold, image_size
                )
                st.session_state["coconexa_analysis"] = {
                    "annotated": annotated,
                    "counts": counts,
                    "unknown_counts": unknown_counts,
                    "detections": detections,
                    "total": len(detections),
                }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption("Check that the uploaded weights are a compatible Ultralytics YOLO model.")

    analysis = st.session_state.get("coconexa_analysis")
    if analysis:
        st.markdown("#### AI detection result")
        st.image(
            analysis["annotated"],
            caption="Each box shows the predicted maturity class and confidence",
            use_container_width=True,
        )
        png_buffer = io.BytesIO()
        Image.fromarray(analysis["annotated"]).save(png_buffer, format="PNG")
        st.download_button(
            "⬇️ Download image with bounding boxes",
            data=png_buffer.getvalue(),
            file_name="coconexa_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

with dashboard_col:
    st.markdown("### 📊 Maturity dashboard")
    analysis = st.session_state.get("coconexa_analysis")
    if analysis:
        counts = analysis["counts"]
        total = analysis["total"]
    else:
        counts = {label: 0 for label in MATURITY_CLASSES}
        total = 0

    st.metric("Total coconuts detected", total)
    for label in MATURITY_CLASSES:
        st.markdown(
            f'<div class="class-row">'
            f'<div class="class-name"><span class="dot" style="background:{CLASS_HEX[label]}"></span>{label}</div>'
            f'<div class="class-count">{counts[label]}</div></div>',
            unsafe_allow_html=True,
        )
    if analysis:
        st.caption("Counts update from the model's accepted detections for this image.")
        if analysis["unknown_counts"]:
            st.warning(
                "Unmapped model classes: " +
                ", ".join(f"{name} ({count})" for name, count in analysis["unknown_counts"].items()) +
                ". Update normalize_class_name() to map these class names."
            )
        if analysis["detections"]:
            st.markdown("#### Detection list")
            st.dataframe(analysis["detections"], use_container_width=True, hide_index=True)
        else:
            st.info("No coconuts detected. Try a clearer image or lower the confidence threshold.")
        st.markdown("#### 🔊 Voice assistance")
        st.caption("Generate an audio summary of the counts.")
        if st.button("Speak analysis results", use_container_width=True):
            try:
                audio_bytes, spoken_text = make_voice_summary(counts, total, voice_language)
                st.session_state["coconexa_voice"] = audio_bytes
                st.session_state["coconexa_voice_text"] = spoken_text
            except Exception as exc:
                st.error(
                    "Could not create voice audio. Check network access and gTTS language support. "
                    f"Details: {exc}"
                )
        if st.session_state.get("coconexa_voice"):
            st.audio(st.session_state["coconexa_voice"], format="audio/mp3")
            with st.expander("Read spoken summary"):
                st.write(st.session_state.get("coconexa_voice_text", ""))
    else:
        st.caption("Your class counts will appear here after you analyze an image.")
        for label in MATURITY_CLASSES:
            st.markdown(
                f'<div class="class-row">'
                f'<div class="class-name"><span class="dot" style="background:{CLASS_HEX[label]}"></span>{label}</div>'
                f'<div class="class-count">—</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("#### 🔊 Voice assistance")
        st.caption("After analysis, play a spoken summary in English or Malayalam.")

st.divider()
st.caption(
    "CocoNexa is a decision-support prototype. Counts represent coconuts detected in the selected image, "
    "not necessarily the total number on a tree. Model predictions should be validated against labeled test images."
)

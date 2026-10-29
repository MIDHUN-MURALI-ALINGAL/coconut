# CocoNexa — Coconut Maturity Analyzer

Streamlit website for coconut detection, maturity-wise counting, bounding boxes, and optional voice summaries. Designed for Render deployment.

## Features

- Upload an image or take a picture with a device camera.
- Dashboard beside the image showing total coconuts and counts for:
  - Young
  - Semi-mature
  - Mature
  - Over mature
- Individual colour-coded bounding boxes with maturity labels and confidence scores.
- Adjustable confidence threshold, IoU, and inference size.
- Download the annotated result image.
- Voice summary in English or Malayalam using gTTS.

## Important: supply your trained model

This project does not include model weights and does not fabricate detections. For real analysis, provide a compatible Ultralytics YOLO `.pt` model trained to detect coconuts with maturity-specific classes.

By default the app looks for `best.pt` in the repository root. You can alternatively upload `.pt` weights from the sidebar or set `COCONEXA_MODEL_PATH` to another model file.

Recommended class names:

- `Young`
- `Semi-mature`
- `Mature`
- `Over mature`

The app maps some common aliases such as `immature` to `Young`. If your model uses different class names, edit `normalize_class_name()` in `app.py`. A model trained with only one generic `coconut` class cannot infer all four maturity stages reliably without maturity-specific training or a separate classifier.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows Command Prompt
.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Render

1. Upload `app.py`, `requirements.txt`, `render.yaml`, `.gitignore`, and `README.md` to a GitHub repository.
2. Add `best.pt` if its size and your repository policy allow it. Otherwise, upload weights through the app sidebar after it starts. For large models, use an appropriate secure storage option.
3. In Render, choose **New + → Blueprint**, connect the repository, and let Render read `render.yaml`.
4. Wait for deployment, then open the generated `onrender.com` URL.
5. Upload or capture a coconut image and press **Analyze coconuts**.

### Render settings

Build command: `pip install -r requirements.txt`

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Voice assistance

Voice summaries are generated on demand by gTTS and require outbound network access. Malayalam voice quality and availability depend on the TTS service. If voice generation fails on a deployed instance, the dashboard and image analysis can still work.

## Limitations

- The count is the number of detections in the uploaded image, not the total yield on a tree.
- Occlusion, lighting, duplicate detections, and incorrect training labels can affect results.
- Browser camera access requires permission and generally works on HTTPS.
- Validate model performance against a separate, labeled test dataset before using predictions for harvest decisions.

# CocoNexa — Coconut Maturity Analyzer

A Streamlit web app for coconut detection, counting, and maturity classification using a trained Ultralytics YOLO model. Designed for deployment on Render.

## Features

- Upload a coconut image (JPG/PNG).
- Draw a bounding box around every detected coconut.
- Label each box with the predicted maturity and confidence score.
- Show total detections and counts for Young, Semi-mature, Mature, and Over mature.
- Adjust confidence, NMS IoU, and inference image size.
- View detection details and download the annotated image.

## Important: add your trained model

This repository does **not** include model weights or fabricate predictions. Your trained YOLO model must be trained to detect coconuts with maturity-specific classes.

The app expects `best.pt` in the repository root by default. You can instead set `COCONEXA_MODEL_PATH` to a different model path, or upload `.pt` weights through the sidebar after the app starts.

Recommended class names in the trained model:

- `Young`
- `Semi-mature`
- `Mature`
- `Over mature`

The app maps common aliases such as `immature` to `Young`. If your model uses other class names, edit `normalize_class_name()` in `app.py`. A detector trained with only a generic `coconut` class cannot reliably produce four maturity categories without a suitable maturity classifier/model.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal.

## Deploy to Render

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `render.yaml`, `.gitignore`, and this README.
2. Add your trained model weights. For a small model, you can commit `best.pt` if the file size and your repository policies permit it. Otherwise store weights externally and download them during the build using a secure, private URL or use the in-app model upload. Do not commit secrets or credentials.
3. In Render, select **New + → Blueprint** and connect the GitHub repository. Render will read `render.yaml`.
4. Wait for the build to finish and open the generated `onrender.com` URL.
5. If you did not package `best.pt`, upload the `.pt` file in the sidebar before analyzing an image.

### Render notes

- YOLO dependencies and model weights can make the initial build and startup slower.
- Free web services may sleep when idle; the first request after sleeping may be slow.
- Check the maximum file size and memory available on your chosen Render plan for the model you use.
- Do not expose private model download URLs or tokens in public source code.

## Model evaluation

Before using results for harvesting decisions, evaluate the model on separate labeled test images. Detection and counting can be affected by occlusion, lighting, duplicate detections, and maturity labels that are visually similar. Counts represent visible coconuts detected in the uploaded image, not necessarily the total yield on the tree.

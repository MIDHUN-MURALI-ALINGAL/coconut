# CocoNexa — Render Deployment

This is the Streamlit web application package for CocoNexa.

## Features
- Upload a coconut image or take a picture with the browser camera.
- YOLO object detection with individual bounding boxes and confidence labels.
- Counts for Young, Semi-mature, Mature, and Over mature.
- Detection table and annotated-image download.
- Optional English/Malayalam spoken summary using gTTS (internet connection required).

## Important: model weights are not included
You must train a compatible YOLO object-detection model using the supplied Colab training notebook and obtain `best.pt`. The model must have correctly labeled bounding boxes for all four classes:
0 Young
1 Semi-mature
2 Mature
3 Over mature

Without compatible weights, the website will deploy but cannot produce real detections. Do not rename a classification-only model to `best.pt`; it must be an object-detection model.

## Deploy on Render
1. Extract this ZIP.
2. Create a GitHub repository and upload the files from this ZIP to the repository root (`app.py`, `requirements.txt`, `render.yaml`, `README.md`, `.gitignore`).
3. Add your trained `best.pt` file to the repository root if its size and your GitHub setup allow it. Otherwise, deploy without it and use the sidebar model uploader after opening the website, or configure external model storage.
4. In Render, choose **New → Blueprint** and connect the GitHub repository containing `render.yaml`, or create a Python web service with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy. Open the public URL and grant camera permission if using “Take a picture”.

## Model class names
The app recognizes common aliases such as `immature` → Young and `semi mature` → Semi-mature. Best results require the model's actual class names to be exactly the four CocoNexa names in the order above.

## Notes
- gTTS requires network access to its speech service. If it cannot connect, image detection can still work.
- Free Render instances may sleep when idle and can have limited memory. YOLO inference may be slow or fail for large models on limited memory.
- Validate performance on held-out coconut photos before relying on counts or maturity predictions.

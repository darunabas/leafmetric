import os
import base64
import json
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify, make_response
from ultralytics import YOLO

# Download model if needed
MODEL_URL = "https://www.dropbox.com/scl/fi/txdy81ui02rzxmfu1fghw/best.pt?rlkey=k8cpkoe2qur9is5wr2j1erwbx&dl=1"
MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        f.write(requests.get(MODEL_URL).content)

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_leaf_areas(result, img_shape):
    height, width = img_shape[:2]
    sheet_height_cm = 41.5
    sheet_width_cm = 29.0

    pixel_to_cm_x = sheet_width_cm / width
    pixel_to_cm_y = sheet_height_cm / height
    pixel_area_to_cm2 = pixel_to_cm_x * pixel_to_cm_y

    areas_cm2 = []

    if result.masks is not None:
        for mask_tensor in result.masks.data:
            mask = mask_tensor.cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area_pixels = cv2.contourArea(cnt)
                area_cm2 = area_pixels * pixel_area_to_cm2
                if area_cm2 > 0.1:  # optional filter for noise
                    areas_cm2.append(round(area_cm2, 2))

    return areas_cm2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')  # Handle multiple files
    threshold = request.form.get('threshold', default=0.7, type=float)

    if not files:
        return jsonify({'error': 'No files uploaded'})

    responses = []

    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            results = model.predict(source=file_path, conf=threshold, save=False)
            result = results[0]

            if result.masks is None or result.masks.data is None:
                responses.append({
                    'filename': file.filename,
                    'error': 'No leaf area detected.'
                })
                continue

            overlay_image = result.plot(boxes=False, masks=True)
            overlay_path = os.path.join(OUTPUT_FOLDER, "overlay_" + file.filename)
            cv2.imwrite(overlay_path, overlay_image)

            with open(overlay_path, "rb") as img_f:
                encoded_image = base64.b64encode(img_f.read()).decode("utf-8")

            image_shape = cv2.imread(file_path).shape
            areas = calculate_leaf_areas(result, image_shape)

            responses.append({
                'image': encoded_image,
                'filename': file.filename,
                'areas': areas
            })

        except Exception as e:
            responses.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify(responses)

@app.route('/recalculate', methods=['POST'])
def recalculate():
    filename = request.form.get('filename')
    threshold = float(request.form.get('threshold', 0.7))

    if not filename:
        return jsonify({'error': 'No image filename provided.'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'Image not found on server.'})

    try:
        results = model.predict(source=file_path, conf=threshold, save=False)
        result = results[0]

        if result.masks is None or result.masks.data is None:
            return jsonify({'error': 'No leaf area detected at this threshold.'})

        overlay_image = result.plot(boxes=False, masks=True)
        overlay_path = os.path.join(OUTPUT_FOLDER, "overlay_" + filename)
        cv2.imwrite(overlay_path, overlay_image)

        with open(overlay_path, "rb") as img_f:
            encoded_image = base64.b64encode(img_f.read()).decode("utf-8")

        # Calculate leaf areas
        image_shape = cv2.imread(file_path).shape
        areas = calculate_leaf_areas(result, image_shape)

        return jsonify({
            'image': encoded_image,
            'areas': areas
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
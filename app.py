import os
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import SRModel
import uuid

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Model (Global for efficiency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRModel(upscale_factor=2).to(device)
MODEL_PATH = "model_new.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded successfully on {device}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please ensure training is complete.")

def enhance_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_input = img_input.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(img_input)
        output = torch.clamp(output, 0, 1)

    output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_img = (output_img * 255.0).astype(np.uint8)
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_img_bgr)
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(RESULT_FOLDER, filename)
    
    file.save(input_path)
    
    if enhance_image(input_path, output_path):
        return jsonify({
            'input_url': f'/static/uploads/{filename}',
            'output_url': f'/static/results/{filename}'
        })
    else:
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == '__main__':
    # Listen on all interfaces (0.0.0.0) so team members can access via network IP
    app.run(host='0.0.0.0', port=5000, debug=False)

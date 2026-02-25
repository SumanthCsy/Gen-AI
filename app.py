import os
import torch
import cv2
import numpy as np
import base64
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from model import SRModel

app = Flask(__name__)
CORS(app)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRModel(upscale_factor=2).to(device)
MODEL_PATH = "model_new.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded successfully on {device}")
else:
    print(f"Warning: {MODEL_PATH} not found.")

def enhance_image_in_memory(image_bytes):
    # Decode image from memory
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_input = img_input.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(img_input)
        output = torch.clamp(output, 0, 1)

    output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_img = (output_img * 255.0).astype(np.uint8)
    
    # Encode result to base64 string
    retval, buffer = cv2.imencode('.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"

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

    # Process entirely in memory - NO DISK STORAGE used for images
    image_bytes = file.read()
    
    # Convert input to base64 for preview
    base64_input = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    
    # Process through Neural Network
    enhanced_base64 = enhance_image_in_memory(image_bytes)
    
    if enhanced_base64:
        # Return base64 strings directly to the browser
        return jsonify({
            'input_url': base64_input,
            'output_url': enhanced_base64
        })
    else:
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == '__main__':
    # Listen on all interfaces for network hosting
    app.run(host='0.0.0.0', port=5000, debug=False)

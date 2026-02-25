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
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Could not decode image bytes.")
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        
        output = None
        try:
            # Try GPU
            print(f"Attempting GPU inference (Device: {device})...")
            # Ensure model is on the correct device
            model.to(device)
            img_input_gpu = img_input.unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model(img_input_gpu)
                output = torch.clamp(output, 0, 1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU OOM: Falling back to CPU...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.to("cpu")
                img_input_cpu = img_input.unsqueeze(0).to("cpu")
                with torch.inference_mode():
                    output = model(img_input_cpu)
                    output = torch.clamp(output, 0, 1)
                # Move back to GPU for future requests
                model.to(device)
            else:
                print(f"RuntimeError during inference: {e}")
                model.to(device) # ensure we try to go back
                raise e
        
        if output is None:
            return None

        output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_img = (output_img * 255.0).astype(np.uint8)
        
        retval, buffer = cv2.imencode('.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return f"data:image/png;base64,{base64_img}"
    except Exception as e:
        print(f"Neural Network Error: {e}")
        import traceback
        traceback.print_exc()
        # Safe recovery: move model back to default device
        try:
            model.to(device)
        except:
            pass
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        print(f"Received image: {file.filename}")
        # Process entirely in memory - NO DISK STORAGE used for images
        image_bytes = file.read()
        print(f"Read {len(image_bytes)} bytes")
        
        # Convert input to base64 for preview
        try:
            base64_input = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        except Exception as e:
            print(f"Error encoding input: {e}")
            return jsonify({'error': 'Input encoding failed'}), 500
        
        # Process through Neural Network
        print("Starting enhancement...")
        enhanced_base64 = enhance_image_in_memory(image_bytes)
        
        if enhanced_base64:
            print("Enhancement successful, returning result.")
            # Return base64 strings directly to the browser
            return jsonify({
                'input_url': base64_input,
                'output_url': enhanced_base64
            })
        else:
            print("Enhancement returned None.")
            return jsonify({'error': 'Processing failed inside model'}), 500
    except Exception as e:
        print(f"General App Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Listen on all interfaces for network hosting
    app.run(host='0.0.0.0', port=5000, debug=False)

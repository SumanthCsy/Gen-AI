import torch
import cv2
import numpy as np
import os
from model import SRModel

def enhance_image(image_path, model_path="model_new.pth", output_path="output.png"):
    # Re-enable GPU for instant results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")
    
    # Load Model
    model = SRModel(upscale_factor=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    model.eval()

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    img_input = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_input = img_input.unsqueeze(0).to(device) 

    # Run Inference (Using optimized inference mode)
    with torch.inference_mode():
        output = model(img_input)
        output = torch.clamp(output, 0, 1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Free VRAM immediately

    # Postprocess
    output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_img = (output_img * 255.0).astype(np.uint8)
    
    # Convert to BGR for saving
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(output_path, output_img_bgr)
    print(f"Enhanced image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    # You can change this to any image path you want to test
    test_image = "testimage.png" 
    
    # Create a dummy LR image if it doesn't exist for demonstration
    if not os.path.exists(test_image):
        print(f"Creating a dummy image for testing: {test_image}")
        dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(test_image, dummy)

    enhance_image(test_image)

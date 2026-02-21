import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import cv2
import io
import base64
from torchvision import transforms
from .explainability import XAIExplainer

class DeepFraudDetector:
    def __init__(self, model_name="vit_tiny_patch16_224", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Deep Learning Detector on {self.device}...")
        
        # Load pre-trained model
        # Using a tiny ViT for performance since we are doing sliding window on CPU/low-end GPU
        self.model = timm.create_model(model_name, pretrained=True, num_classes=2)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize XAI Explainer
        self.explainer = XAIExplainer(self.model)

    def sliding_window_inference(self, image_path, patch_size=256, stride=128):
        """
        Performs patch-based inference to detect localized tampering.
        """
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        
        # Initialize score map
        # We'll use a smaller grid and then upscale for efficiency
        cols = (w - patch_size) // stride + 1
        rows = (h - patch_size) // stride + 1
        
        if cols <= 0 or rows <= 0:
            # Image smaller than patch size, just run once on the whole thing (resized)
            return self.single_inference(img), 0.5

        heatmap_grid = np.zeros((rows, cols))
        
        with torch.no_grad():
            for i in range(rows):
                for j in range(cols):
                    left = j * stride
                    top = i * stride
                    right = left + patch_size
                    bottom = top + patch_size
                    
                    patch = img.crop((left, top, right, bottom))
                    input_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    # Use class 1 as "forgery" probability
                    forgery_prob = probs[0][1].item()
                    heatmap_grid[i, j] = forgery_prob

        # Average probability across all patches for the combined score
        avg_score = float(np.mean(heatmap_grid))
        
        # Generate high-resolution heatmap
        heatmap_resized = cv2.resize(heatmap_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Convert to RGB for PIL/Base64
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb), avg_score

    def single_inference(self, pil_img):
        """Fallback for small images"""
        with torch.no_grad():
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs[0][1].item()

    def generate_explanation(self, image_path):
        """
        Generates a Grad-CAM explanation image for the whole document.
        """
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        # Enable gradients for Grad-CAM
        input_tensor.requires_grad = True
        
        explanation_img = self.explainer.generate_explanation(input_tensor, img)
        return explanation_img

def dl_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Singleton instance
dl_detector = DeepFraudDetector()

import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import io
import base64

class XAIExplainer:
    def __init__(self, model, target_layers=None):
        self.model = model
        # For ViT (timm), the target layer is usually model.blocks[-1].norm1 or similar
        # For CNNs, it's usually the last convolutional layer
        if target_layers is None:
            if hasattr(model, 'blocks'): # ViT
                self.target_layers = [model.blocks[-1]]
            elif hasattr(model, 'conv_head'): # EfficientNet
                self.target_layers = [model.conv_head]
            else:
                # Fallback: try to find the last conv/block
                self.target_layers = [list(model.children())[-2]] 
        else:
            self.target_layers = target_layers
            
        self.cam = GradCAM(model=model, target_layers=self.target_layers)

    def generate_explanation(self, input_tensor, original_image_pil, target_category=1):
        """
        Generates Grad-CAM visualization for the specified category.
        """
        # input_tensor is (1, 3, 224, 224)
        targets = [ClassifierOutputTarget(target_category)]
        
        # Calculate grayscale CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # grayscale_cam is (1, 224, 224)
        grayscale_cam = grayscale_cam[0, :]
        
        # Prepare original image for overlay (resized to 224, 224 for Grad-CAM)
        img_np = np.array(original_image_pil.resize((224, 224))) / 255.0
        
        # Overlay heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Resize visualization back to original high-res size for the UI
        w, h = original_image_pil.size
        visualization_high_res = cv2.resize(visualization, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return Image.fromarray(visualization_high_res)

def xai_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

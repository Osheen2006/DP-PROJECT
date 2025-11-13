import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationModels:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all models from disk."""
        try:
            self.models["segformer"] = torch.load("C:\\Users\\hiosh\\Downloads\\Microsoft VS Code\\DP project\\segformer.pth", map_location=self.device)
            self.models["unetpp"] = torch.load("C:\\Users\\hiosh\\Downloads\\Microsoft VS Code\\DP project\\unet++.pth", map_location=self.device)
            self.models["unet"] = torch.load("C:\\Users\\hiosh\\Downloads\\Microsoft VS Code\\DP project\\unet.pth", map_location=self.device)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")

        for m in self.models.values():
            m.eval()

    def preprocess_image(self, image_path):
    
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0).to(self.device)

    def segment(self, image_path, model_name="unet"):
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded.")
        model = self.models[model_name]

        img_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

    
        return Image.fromarray(mask)

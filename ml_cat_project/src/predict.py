# src/predict.py
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from PIL import Image

class_names = ["cat", "not_cat"]

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_image(model: nn.Module, image_path: str) -> Any:
    image = Image.open(image_path).convert("RGB")
    img_t = predict_transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = outputs.max(1)
    return class_names[predicted.item()]

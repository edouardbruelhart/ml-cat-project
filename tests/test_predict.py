# tests/test_predict.py
import torch

from ml_cat_project.src.model import SimpleNN  # your model class
from ml_cat_project.src.predict import class_names, predict_image


def test_predict_image():
    # 1. Create a dummy model that outputs zeros (always predicts first class)
    model = SimpleNN()

    # 2. Create a fake image (3 channels, 224x224)
    dummy_image = torch.zeros((3, 224, 224))

    # 3. Save dummy image temporarily
    from PIL import Image

    dummy_pil = Image.fromarray((dummy_image.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    dummy_pil.save("dummy_cat.jpg")

    # 4. Predict
    label = predict_image(model, "dummy_cat.jpg")

    # 5. Check if label is in class_names
    assert label in class_names

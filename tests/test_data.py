import os

from ml_cat_project.src.data import get_dataloaders


def test_dataloader():
    data_dir = "ml_cat_project/toy_dataset/raw" if os.getenv("CI") == "true" else "ml_cat_project/dataset/raw"
    loader, classes = get_dataloaders(data_dir=data_dir)
    images, labels = next(iter(loader))

    # Check image shape: batch of 16, RGB, 224x224
    assert images.shape == (16, 3, 224, 224)

    # Check labels match batch size
    assert labels.shape[0] == 16

    # Check that class names exist
    assert set(classes) == {"cat", "not_cat"}

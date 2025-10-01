from ml_cat_project.src.data import get_dataloaders


def test_dataloader_runs():
    loader, classes = get_dataloaders()
    batch = next(iter(loader))
    images, _ = batch
    assert images.shape[1:] == (3, 224, 224)  # 3-channel RGB, 224x224
    assert len(classes) == 2  # cat / not-cat

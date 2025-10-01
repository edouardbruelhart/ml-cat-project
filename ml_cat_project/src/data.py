import torchvision.transforms as transforms  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # type: ignore[import-untyped]


def get_dataloaders(data_dir: str = "ml_cat_project/dataset/raw", batch_size: int = 16) -> tuple[DataLoader, list[str]]:
    """
    Returns PyTorch dataloaders for training.
    Uses data augmentation for training and only normalization for validation/test.
    In CI (GitHub Actions), falls back to a dummy dataset.
    """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Otherwise load real dataset
    dataset = ImageFolder(data_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes

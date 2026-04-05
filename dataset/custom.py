import torchvision.transforms.v2 as transforms
from torchvision import datasets
from torch.utils.data import random_split


def get_custom_data(train_path: str, augment="False", split_ratio: float = 0.8):
    IMAGE_SIZE = 224
    if augment == "True":
        train_tfms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        test_tfms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        test_tfms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

    full_dataset = datasets.ImageFolder(train_path, transform=train_tfms)
    total_size = len(full_dataset)
    train_size = int(split_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_tfms
    val_dataset.dataset.transform = test_tfms

    return train_dataset, val_dataset

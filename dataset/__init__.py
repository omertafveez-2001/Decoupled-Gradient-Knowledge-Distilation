import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader

from .mnist import get_MNIST_dataset
from .cifar import get_CIFAR10_dataset, get_CIFAR100_dataset
from .svhn import get_SVHN_dataset
from .food101 import get_Food101_dataset
from .tiny_imagenet import TinyImageNet, get_TinyImageNet_dataset
from .custom import get_custom_data
from .augmentations import (
    AddNoiseToPatch,
    PatchScrambler,
    get_noised_data,
    get_scrambled_data,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

# Native image sizes — CIFAR/SVHN/MNIST are 32x32, TinyImageNet is 64x64
DATASET_SIZE = {
    "MNIST": 32,
    "CIFAR-10": 32,
    "CIFAR-100": 32,
    "SVHN": 32,
    "Food101": 224,
    "TinyImageNet": 64,
}


def _build_transforms(augment: bool, mean, std, image_size: int):
    train_tfms = transforms.Compose(
        [transforms.Resize((image_size, image_size))]
        + ([transforms.RandomHorizontalFlip()] if augment else [])
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfms, test_tfms


def get_dataset(name: str, augment: bool = False, root: str = "./data"):
    image_size = DATASET_SIZE.get(name, 224)

    if name == "TinyImageNet":
        train_tfms, test_tfms = _build_transforms(augment, IMAGENET_MEAN, IMAGENET_STD, image_size)
        return get_TinyImageNet_dataset(root, train_tfms, test_tfms)

    train_tfms, test_tfms = _build_transforms(augment, CIFAR_MEAN, CIFAR_STD, image_size)

    if name == "MNIST":
        return get_MNIST_dataset(root, train_tfms, test_tfms)
    elif name == "CIFAR-10":
        return get_CIFAR10_dataset(root, train_tfms, test_tfms)
    elif name == "CIFAR-100":
        return get_CIFAR100_dataset(root, train_tfms, test_tfms)
    elif name == "SVHN":
        return get_SVHN_dataset(root, train_tfms, test_tfms)
    elif name == "Food101":
        return get_Food101_dataset(root, train_tfms, test_tfms)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: MNIST, CIFAR-10, CIFAR-100, SVHN, Food101, TinyImageNet.")


def get_dataloader(dataset: Dataset, batch_size: int, is_train: bool, num_workers: int = 1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

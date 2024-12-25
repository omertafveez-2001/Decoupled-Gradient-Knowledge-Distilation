import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torchvision import datasets
from torch.utils.data import random_split

def get_custom_data(train_path: str, TRAIN_TFMS, TEST_TFMS, split_ratio: float = 0.8):
    # Load the dataset
    full_dataset = datasets.ImageFolder(train_path, transform=TRAIN_TFMS)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(split_ratio * total_size)
    val_size = total_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transformations
    train_dataset.dataset.transform = TRAIN_TFMS
    val_dataset.dataset.transform = TEST_TFMS
    
    return train_dataset, val_dataset


def get_dataset(name, augment=False, root="./data"):
    IMAGE_SIZE = 224
    if augment:
        TRAIN_TFMS = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        TEST_TFMS = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    else:
        TRAIN_TFMS = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        TEST_TFMS = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

    if name == "MNIST":
        return get_MNIST_dataset(root, TRAIN_TFMS, TEST_TFMS)
    elif name == "CIFAR-10":
        return get_CIFAR10_dataset(root, TRAIN_TFMS, TEST_TFMS)
    elif name == "CIFAR-100":
        return get_CIFAR100_dataset(root, TRAIN_TFMS, TEST_TFMS)
    elif name == "SVHN":
        return get_SVHN_dataset(root, TRAIN_TFMS, TEST_TFMS)
    elif name == "Food101":
        return get_Food101_dataset(root, TRAIN_TFMS, TEST_TFMS)
    else:
        raise ValueError("Received invalid dataset name - please check data.py")


def get_dataloader(
    dataset: Dataset, batch_size: int, is_train: bool, num_workers: int = 1
):

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers
    )
    return loader


def get_MNIST_dataset(root: str, TRAIN_TFMS, TEST_TFMS):

    trainset = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset


def get_CIFAR10_dataset(root: str, TRAIN_TFMS, TEST_TFMS):

    trainset = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset


def get_CIFAR100_dataset(root: str, TRAIN_TFMS, TEST_TFMS):

    trainset = torchvision.datasets.CIFAR100(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR100(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset


def get_SVHN_dataset(root: str, TRAIN_TFMS, TEST_TFMS):

    trainset = torchvision.datasets.SVHN(
        root, split="train", download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.SVHN(
        root, split="test", download=True, transform=TEST_TFMS
    )

    return trainset, testset


def get_Food101_dataset(root: str, TRAIN_TFMS, TEST_TFMS):
    trainset = torchvision.datasets.Food101(
        root, split="train", download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.Food101(
        root, split="test", download=True, transform=TEST_TFMS
    )

    return trainset, testset

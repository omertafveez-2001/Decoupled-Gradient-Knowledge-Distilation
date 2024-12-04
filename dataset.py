import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Optional


IMAGE_SIZE = 224
TRAIN_TFMS = transforms.Compose([
    transforms.Resize((224, 2224)),
    transforms.ToTensor(),
])
TEST_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_dataset(name):    
    if name == 'MNIST':
        return get_MNIST_dataset
    elif name == 'CIFAR-10':
        return get_CIFAR10_dataset
    elif name == 'CIFAR-100':
        return get_CIFAR100_dataset
    elif name == 'SVHN':
        return get_SVHN_dataset
    else:
        raise ValueError("Received invalid dataset name - please check data.py")
    
def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   is_train: bool,
                   num_workers: int = 1):
    
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=is_train, num_workers=num_workers)
    return loader


def get_MNIST_dataset(root: str):

    trainset = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_CIFAR10_dataset(root: str):

    trainset = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_CIFAR100_dataset(root: str):

    trainset = torchvision.datasets.CIFAR100(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR100(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset


def get_SVHN_dataset(root: str):

    trainset = torchvision.datasets.SVHN(
        root, split='train', download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.SVHN(
        root, split='test', download=True, transform=TEST_TFMS
    )

    return trainset, testset
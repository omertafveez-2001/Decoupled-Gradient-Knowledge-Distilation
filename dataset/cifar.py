import torchvision


def get_CIFAR10_dataset(root: str, train_tfms, test_tfms):
    trainset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tfms)
    testset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tfms)
    return trainset, testset


def get_CIFAR100_dataset(root: str, train_tfms, test_tfms):
    trainset = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=train_tfms)
    testset = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=test_tfms)
    return trainset, testset

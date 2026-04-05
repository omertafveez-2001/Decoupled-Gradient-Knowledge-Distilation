import torchvision


def get_SVHN_dataset(root: str, train_tfms, test_tfms):
    trainset = torchvision.datasets.SVHN(root, split="train", download=True, transform=train_tfms)
    testset = torchvision.datasets.SVHN(root, split="test", download=True, transform=test_tfms)
    return trainset, testset

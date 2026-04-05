import torchvision


def get_Food101_dataset(root: str, train_tfms, test_tfms):
    trainset = torchvision.datasets.Food101(root, split="train", download=True, transform=train_tfms)
    testset = torchvision.datasets.Food101(root, split="test", download=True, transform=test_tfms)
    return trainset, testset

import torchvision


def get_MNIST_dataset(root: str, train_tfms, test_tfms):
    trainset = torchvision.datasets.MNIST(root, train=True, download=True, transform=train_tfms)
    testset = torchvision.datasets.MNIST(root, train=False, download=True, transform=test_tfms)
    return trainset, testset

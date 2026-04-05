import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.v2 as transforms


class AddNoiseToPatch:
    def __init__(self, noise_level=0.2, patch_coords=(0, 0, 50, 50)):
        self.noise_level = noise_level
        self.patch_coords = patch_coords  # (x1, y1, x2, y2)

    def __call__(self, img):
        img_np = np.array(img)
        x1, y1, x2, y2 = self.patch_coords
        noise = np.random.normal(
            0, self.noise_level, img_np[y1:y2, x1:x2].shape
        ).astype(np.uint8)
        img_np[y1:y2, x1:x2] = np.clip(img_np[y1:y2, x1:x2] + noise, 0, 255)
        return Image.fromarray(img_np)


class PatchScrambler:
    def __init__(self, patch_size=56):
        self.patch_size = patch_size

    def scramble(self, image):
        c, h, w = image.shape
        assert (
            h % self.patch_size == 0 and w % self.patch_size == 0
        ), "Image size must be divisible by patch size"

        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)

        permuted_indices = torch.randperm(patches.size(0))
        scrambled_patches = patches[permuted_indices]

        scrambled_image = (
            scrambled_patches.permute(1, 0, 2, 3)
            .contiguous()
            .view(c, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        )
        scrambled_image = (
            scrambled_image.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
        )
        return scrambled_image

    def __call__(self, image):
        return self.scramble(image)


NOISE_TFMS = lambda noise_size: transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        AddNoiseToPatch(
            noise_level=25, patch_coords=(50, 50, 50 + noise_size, 50 + noise_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

SCRAMBLE_TFMS = lambda patch_size: transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        PatchScrambler(patch_size=patch_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def get_noised_data(name, noise_size, root="./data"):
    tfms = NOISE_TFMS(noise_size)

    if name == "MNIST":
        trainset = torchvision.datasets.MNIST(root + "/train", train=True, download=True, transform=tfms)
        testset = torchvision.datasets.MNIST(root + "/noise-val", train=False, download=True, transform=tfms)
    elif name == "Food101":
        trainset = torchvision.datasets.Food101(root, split="train", download=True, transform=tfms)
        testset = torchvision.datasets.Food101(root, split="test", download=True, transform=tfms)
    elif name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(root + "/train", train=True, download=True, transform=tfms)
        testset = torchvision.datasets.CIFAR10(root + "/noise-val", train=False, download=True, transform=tfms)
    elif name == "CIFAR-100":
        trainset = torchvision.datasets.CIFAR100(root + "/train", train=True, download=True, transform=tfms)
        testset = torchvision.datasets.CIFAR100(root + "/noise-val", train=False, download=True, transform=tfms)
    else:
        raise ValueError(f"Noised data not supported for dataset '{name}'.")

    return trainset, testset


def get_scrambled_data(name, patch_size, root):
    tfms = SCRAMBLE_TFMS(patch_size)

    if name == "MNIST":
        trainset = torchvision.datasets.MNIST(root + "/train", train=True, download=True, transform=tfms)
        testset = torchvision.datasets.MNIST(root + "/scrambled-val", train=False, download=True, transform=tfms)
    elif name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(root + "/train", train=True, download=True, transform=tfms)
        testset = torchvision.datasets.CIFAR10(root + "/scrambled-val", train=False, download=True, transform=tfms)
    elif name == "CIFAR-100":
        trainset = torchvision.datasets.CIFAR100(root + "/train", train=True, download=True, transform=tfms)
        testset = torchvision.datasets.CIFAR100(root + "/scrambled-val", train=False, download=True, transform=tfms)
    elif name == "Food101":
        trainset = torchvision.datasets.Food101(root, split="train", download=True, transform=tfms)
        testset = torchvision.datasets.Food101(root, split="test", download=True, transform=tfms)
    else:
        raise ValueError(f"Scrambled data not supported for dataset '{name}'.")

    return trainset, testset

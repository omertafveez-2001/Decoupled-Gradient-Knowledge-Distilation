import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torchvision import datasets
from torch.utils.data import random_split


class AddNoiseToPatch:
    def __init__(self, noise_level=0.2, patch_coords=(0, 0, 50, 50)):
        self.noise_level = noise_level
        self.patch_coords = patch_coords  # (x1, y1, x2, y2)

    def __call__(self, img):
        # Convert to numpy array
        img_np = np.array(img)

        # Extract patch coordinates
        x1, y1, x2, y2 = self.patch_coords

        # Generate random noise
        noise = np.random.normal(
            0, self.noise_level, img_np[y1:y2, x1:x2].shape
        ).astype(np.uint8)

        # Add noise to the patch
        img_np[y1:y2, x1:x2] = np.clip(img_np[y1:y2, x1:x2] + noise, 0, 255)

        # Convert back to PIL Image
        return Image.fromarray(img_np)


class PatchScrambler:
    def __init__(self, patch_size=56):
        self.patch_size = patch_size

    def scramble(self, image):
        c, h, w = image.shape

        # Checl if image is divisible by patch_size
        assert (
            h % self.patch_size == 0 and w % self.patch_size == 0
        ), "Image size must be divisible by patch size"

        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        # Split image into patches
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )

        # Reshape into (num_patches_h * num_patches_w, C, patch_size, patch_size)
        patches = patches.contiguous().view(c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)

        # Shuffle the patches
        permuted_indices = torch.randperm(patches.size(0))
        scrambled_patches = patches[permuted_indices]

        # Reshape back into original image form
        scrambled_image = (
            scrambled_patches.permute(1, 0, 2, 3)
            .contiguous()
            .view(c, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        )

        # Reassemble the image from scrambled patches
        scrambled_image = (
            scrambled_image.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
        )

        return scrambled_image

    def __call__(self, image):
        return self.scramble(image)


def get_noised_data(name, noise_size, root="./data"):

    NOISE_TEST_TFMS = transforms.Compose(
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

    if name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root + "/train", train=True, download=True, transform=NOISE_TEST_TFMS
        )

        testset = torchvision.datasets.MNIST(
            root + "/noise-val", train=False, download=True, transform=NOISE_TEST_TFMS
        )

    if name == "Food101":
        trainset = torchvision.datasets.Food101(
            root, split="train", download=True, transform=NOISE_TEST_TFMS
        )

        testset = torchvision.datasets.Food101(
            root, split="test", download=True, transform=NOISE_TEST_TFMS
        )

    elif name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(
            root + "/train", train=True, download=True, transform=NOISE_TEST_TFMS
        )

        testset = torchvision.datasets.CIFAR10(
            root + "/noise-val", train=False, download=True, transform=NOISE_TEST_TFMS
        )

    elif name == "CIFAR-100":
        trainset = torchvision.datasets.CIFAR100(
            root + "/train", train=True, download=True, transform=NOISE_TEST_TFMS
        )

        testset = torchvision.datasets.CIFAR100(
            root + "/noise-val", train=False, download=True, transform=NOISE_TEST_TFMS
        )

    else:
        raise ValueError(
            "Incorrect dataset name. Choose from [MNIST, CIFAR-10, CIFAR-100]."
        )

    return trainset, testset


def get_scrambled_data(name, patch_size, root):

    SCRAMBLE_TFMS = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            PatchScrambler(patch_size=patch_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root + "/train", train=True, download=True, transform=SCRAMBLE_TFMS
        )

        testset = torchvision.datasets.MNIST(
            root + "/scrambled-val", train=False, download=True, transform=SCRAMBLE_TFMS
        )

    elif name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(
            root + "/train", train=True, download=True, transform=SCRAMBLE_TFMS
        )

        testset = torchvision.datasets.CIFAR10(
            root + "/scrambled-val", train=False, download=True, transform=SCRAMBLE_TFMS
        )

    elif name == "CIFAR-100":
        trainset = torchvision.datasets.CIFAR100(
            root + "/train", train=True, download=True, transform=SCRAMBLE_TFMS
        )

        testset = torchvision.datasets.CIFAR100(
            root + "/scrambled-val", train=False, download=True, transform=SCRAMBLE_TFMS
        )

    elif name == "Food101":
        trainset = torchvision.datasets.Food101(
            root, split="train", download=True, transform=SCRAMBLE_TFMS
        )

        testset = torchvision.datasets.Food101(
            root, split="test", download=True, transform=SCRAMBLE_TFMS
        )

    else:
        raise ValueError(
            "Incorrect dataset name. Choose from [MNIST, CIFAR-10, CIFAR-100]."
        )

    return trainset, testset


def get_custom_data(train_path: str, augment=False, split_ratio: float = 0.8):
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
    elif name == "TinyImageNet":
        return get_TinyImageNet_dataset(root, augment=augment)
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


def get_TinyImageNet_dataset(root: str, augment=False):
    IMAGE_SIZE = 224
    if augment:
        TRAIN_TFMS = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        TEST_TFMS = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

    trainset = TinyImageNet(root, split="train", transform=TRAIN_TFMS)
    testset = TinyImageNet(root, split="val", transform=TEST_TFMS)

    return trainset, testset


class TinyImageNet(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.data = []
        self.labels = []
        self.label_map = self._create_label_map()

        self._load_data()

    def _create_label_map(self):
        label_map = {}
        with open(os.path.join(self.root, "wnids.txt"), "r") as f:
            for idx, line in enumerate(f.readlines()):
                label_map[line.strip()] = idx
        return label_map

    def _load_data(self):
        if self.split == "train":
            for label in self.label_map.keys():
                label_dir = os.path.join(self.root, "train", label, "images")
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(self.label_map[label])
        elif self.split == "val":
            val_img_dir = os.path.join(self.root, "val", "images")
            with open(os.path.join(self.root, "val", "val_annotations.txt"), "r") as f:
                val_annotations = f.readlines()
            for annotation in val_annotations:
                parts = annotation.split("\t")
                img_name = parts[0]
                img_path = os.path.join(val_img_dir, img_name)
                label = parts[1]
                self.data.append(img_path)
                self.labels.append(self.label_map[label])
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

import os
from PIL import Image
from torch.utils.data import Dataset


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
                    self.data.append(os.path.join(label_dir, img_name))
                    self.labels.append(self.label_map[label])
        elif self.split == "val":
            val_img_dir = os.path.join(self.root, "val", "images")
            with open(os.path.join(self.root, "val", "val_annotations.txt"), "r") as f:
                for annotation in f.readlines():
                    parts = annotation.split("\t")
                    self.data.append(os.path.join(val_img_dir, parts[0]))
                    self.labels.append(self.label_map[parts[1]])
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_TinyImageNet_dataset(root: str, train_tfms, test_tfms):
    trainset = TinyImageNet(root, split="train", transform=train_tfms)
    testset = TinyImageNet(root, split="val", transform=test_tfms)
    return trainset, testset

import torch.nn as nn
from models.resnet import (
    resnet8, resnet14, resnet20, resnet32, resnet44,
    resnet56, resnet110, resnet8x4, resnet32x4,
)
from models.vgg import (
    vgg8, vgg8_bn, vgg11, vgg11_bn, vgg13, vgg13_bn,
    vgg16, vgg16_bn, vgg19, vgg19_bn,
)
from models.ShuffleNetv2 import ShuffleV2

MODEL_REGISTRY = {
    "resnet8":    resnet8,
    "resnet14":   resnet14,
    "resnet20":   resnet20,
    "resnet32":   resnet32,
    "resnet44":   resnet44,
    "resnet56":   resnet56,
    "resnet110":  resnet110,
    "resnet8x4":  resnet8x4,
    "resnet32x4": resnet32x4,
    "vgg8":       vgg8,
    "vgg8_bn":    vgg8_bn,
    "vgg11":      vgg11,
    "vgg11_bn":   vgg11_bn,
    "vgg13":      vgg13,
    "vgg13_bn":   vgg13_bn,
    "vgg16":      vgg16,
    "vgg16_bn":   vgg16_bn,
    "vgg19":      vgg19,
    "vgg19_bn":   vgg19_bn,
    "ShuffleV2":  ShuffleV2,
}


class TeacherModel(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}")
        self.model = MODEL_REGISTRY[model](num_classes=num_classes)

    def forward(self, x):
        out, _ = self.model(x)
        return out


class StudentModel(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}")
        self.model = MODEL_REGISTRY[model](num_classes=num_classes)

    def forward(self, x):
        out, _ = self.model(x)
        return out

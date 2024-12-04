import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, vgg11, vgg13, vgg16, vgg19 


class TeacherModel(nn.Module):
    def __init__(self, model, num_classes):
        super(TeacherModel, self).__init__()

        if model == "resnet18":
            self.model = resnet18(pretrained=True)
        elif model == "resnet34":
            self.model = resnet34(pretrained=True)
        elif model == "resnet50":
            self.model = resnet50(pretrained=True)
        elif model == "resnet101":
            self.model = resnet101(pretrained=True)
        elif model == "vgg11":
            self.model = vgg11(pretrained=True)
        elif model == "vgg13":
            self.model = vgg13(pretrained=True)
        elif model == "vgg16":
            self.model = vgg16(pretrained=True)
        elif model == "vgg19":
            self.model = vgg19(pretrained=True)
        else:
            raise ValueError("Invalid model name")
        
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class StudentModel(nn.Module):
    def __init__(self, model, num_classes):
        super(StudentModel, self).__init__()

        if model == "resnet18":
            self.model = resnet18(pretrained=True)
        elif model == "resnet34":
            self.model = resnet34(pretrained=True)
        elif model == "resnet50":
            self.model = resnet50(pretrained=True)
        elif model == "resnet101":
            self.model = resnet101(pretrained=True)
        elif model == "vgg11":
            self.model = vgg11(pretrained=True)
        elif model == "vgg13":
            self.model = vgg13(pretrained=True)
        elif model == "vgg16":
            self.model = vgg16(pretrained=True)
        elif model == "vgg19":
            self.model = vgg19(pretrained=True)
        else:
            raise ValueError("Invalid model name")
        
        if model.startswith("res"):
            self.model.fc = nn.Linear(self.fc.in_features, num_classes)
        else:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)




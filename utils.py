import random
import numpy as np
import torch
import argparse

def count_parameters(model):
    params = (p.numel() for p in model.parameters() if p.requires_grad)
    # return params in Millions format. e.g. 1.2M
    return sum(params) / 1e6

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Train a student model using knowledge distillation."
    )

    parser.add_argument(
        "--teacher_arch",
        type=str,
        default="resnet18",
        help="Teacher model architecture (e.g. resnet18, resnet50)",
    )

    parser.add_argument(
        "--student_arch",
        type=str,
        default="resnet18",
        help="Student model architecture (e.g. resnet18, mobilenet_v2)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR-10",
        help="Dataset to use (MNIST, CIFAR-10, CIFAR-100, SVHN, Food101, TinyImageNet)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 6.0],
        help="Distillation loss weights [alpha, beta, epsilon]",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Initial learning rate (step decay ÷10 at epochs 150, 180, 210 with 20-epoch linear warmup)",
    )

    parser.add_argument(
        "--augment",
        type=str,
        default=False,
        help="Apply data augmentation during training",
    )

    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Path to a pretrained teacher checkpoint (.pth) to load instead of training",
    )

    parser.add_argument(
        "--student_checkpoint",
        type=str,
        default=None,
        help="Path to a pretrained student checkpoint (.pth) to load instead of training",
    )

    parser.add_argument(
        "--teacher_save_path",
        type=str,
        default=None,
        help="Path to save the trained teacher model (.pth). Skips saving if not provided.",
    )

    parser.add_argument(
        "--student_save_path",
        type=str,
        default=None,
        help="Path to save the trained student model (.pth). Skips saving if not provided.",
    )

    parser.add_argument(
        "--distillation_save_path",
        type=str,
        default=None,
        help="Path to save the distilled student model (.pth). Skips saving if not provided.",
    )

    parser.add_argument(
        "--robustness_eval",
        type=str,
        default=None,
        help="Evaluate robustness under distribution shift: stylized | noised | scrambled",
    )

    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to a custom dataset (ImageFolder format) used when --robustness_eval=stylized",
    )

    parser.add_argument(
        "--distillation_method",
        type=str,
        default=None,
        help="Distillation method: logit_matching | decoupled | decoupled_gradient_distillation",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Temperature for distillation softmax scaling",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup epochs to linearly ramp up the KD loss",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )

    return parser

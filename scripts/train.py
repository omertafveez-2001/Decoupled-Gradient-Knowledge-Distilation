from dataset import get_noised_data, get_scrambled_data, get_custom_data, get_dataset, get_dataloader
from distillation.models import StudentModel, TeacherModel
from distillation.main import KnowledgeDistillation
from scripts.finetune import Finetune
from utils import count_parameters, argument_parser, set_seed
import warnings
import torch
import torch.nn as nn


warnings.filterwarnings("ignore")


if __name__ == "__main__":

    args = argument_parser().parse_args()
    set_seed(args.seed) # Set random seed for reproducibility
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_workers = 2

    # Original Dataset (used for distillation)
    distill_train_ds, distill_test_ds = get_dataset(
        args.dataset, args.augment, root=f"./data/{args.dataset}"
    )

    # Evaluation dataset (may be augmented/biased)
    if args.robustness_eval == "stylized":
        train_ds, test_ds = get_custom_data(args.custom_data_path, args.augment)
    elif args.robustness_eval == "noised":
        train_ds, test_ds = get_noised_data(
            args.dataset, noise_size=100, root=f"./data/{args.dataset}"
        )
    elif args.robustness_eval == "scrambled":
        train_ds, test_ds = get_scrambled_data(
            args.dataset, patch_size=56, root=f"./data/{args.dataset}"
        )
    else:
        train_ds, test_ds = get_dataset(
            args.dataset, args.augment, root=f"./data/{args.dataset}"
        )

    train_loader = get_dataloader(
        train_ds, args.batch_size, is_train=True, num_workers=num_workers
    )
    test_loader = get_dataloader(
        test_ds, args.batch_size, is_train=False, num_workers=num_workers
    )

    distill_train_loader = get_dataloader(
        distill_train_ds, args.batch_size, is_train=True, num_workers=num_workers
    )
    distill_test_loader = get_dataloader(
        distill_test_ds, args.batch_size, is_train=False, num_workers=num_workers
    )

    if args.dataset == "SVHN":
        num_classes = 10
    else:
        num_classes = len(train_ds.classes)

    teacher_model = TeacherModel(args.teacher_arch, num_classes).cuda()
    student_model = StudentModel(args.student_arch, num_classes).cuda()

    print("============================================")
    print(f"Using device: {device}")
    print(f"Teacher Model: {args.teacher_arch} with parameters {count_parameters(teacher_model)}")
    print(f"Student Model: {args.student_arch} with parameters {count_parameters(student_model)}")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Augmented Data: {args.augment}")
    if args.robustness_eval:
        print(f"Robustness Evaluation: {args.robustness_eval}")
    print("============================================")

    criterion = nn.CrossEntropyLoss()

    # Train or load teacher
    if args.teacher_checkpoint:
        print("Loading teacher model from checkpoint...")
        teacher_model.load_state_dict(torch.load(args.teacher_checkpoint))
        print("Teacher model loaded.")
    else:
        print("Finetuning teacher model...")
        teacher_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        teacher = Finetune(
            teacher_model, train_loader, test_loader,
            teacher_optimizer, criterion, device, args.epochs,
            f"{args.teacher_arch}_{args.dataset}", args.dataset, 
        )
        teacher.train()

    # Train student baseline
    if args.student_checkpoint is None:
        print("Finetuning student model...")
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=args.learning_rate)
        student = Finetune(
            student_model, train_loader, test_loader,
            student_optimizer, criterion, device, args.epochs,
            f"{args.student_arch}_{args.dataset}", args.dataset, 
        )
        student.train()

    # Knowledge distillation
    if args.distillation_method:
        print(f"Distilling knowledge using {args.distillation_method}...")
        distill_student = StudentModel(args.student_arch, num_classes).cuda()
        distill_optimizer = torch.optim.AdamW(distill_student.parameters(), lr=args.learning_rate)

        kd = KnowledgeDistillation(
            teacher_model,
            distill_student,
            distill_train_loader,
            distill_test_loader,
            distill_optimizer,
            device,
            args,
            distillation_method=args.distillation_method,
        )
        kd.train()

from dataset import *
from distillation.distillation_process import *
from distillation.decoupled_distillation import *
from finetune import *
import argparse
from utils import *
import os
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a student model using knowledge distillation."
    )
    parser.add_argument(
        "--teachermodel",
        type=str,
        default="resnet18",
        help="Teacher model architecture",
    )
    parser.add_argument(
        "--studentmodel",
        type=str,
        default="resnet18",
        help="Student model architecture",
    )
    parser.add_argument(
        "--dataset", type=str, default="CIFAR-10", help="Dataset to use"
    )
    parser.add_argument(
        "--batchsize", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Number of epochs to [finetune, distill]",
    )
    parser.add_argument(
        "--hyperparameters",
        type=float,
        nargs="+",
        default=[0.5, 0.5, 1, 7, 2, 0.6, 3],
        help="Hyperparameters for distillation loss [alpha, beta, gamma, phi, epsilon, delta, temperature]",
    )
    parser.add_argument(
        "--student_dir",
        type=str,
        default="student",
        help="Directory for saving student model",
    )
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default="teacher",
        help="Directory for saving teacher model",
    )
    parser.add_argument(
        "--distill_dir",
        type=str,
        default="output",
        help="Directory for distillation saving logs and models",
    )
    parser.add_argument(
        "--learningrates",
        type=float,
        nargs="+",
        default=[0.001, 0.1],
        help="Learning rates for [finetune, distill]",
    )
    parser.add_argument("--augment", type=bool, default=False, help="Augment data")
    parser.add_argument(
        "--teachermodel_path",
        type=str,
        default=None,
        help="Path to the finetuned model",
    )
    parser.add_argument(
        "--studentmodel_path",
        type=str,
        default=None,
        help="Path to the finetuned student",
    )
    parser.add_argument(
        "--bias_eval",
        type=str,
        default=None,
        help="Bias Evaluation-choose from style, scrambled, and noised",
    )
    parser.add_argument(
        "--datasetpath",
        type=str,
        default=None,
        help="Custom Dataset for distillation and training",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Choose from logit_matching, decoupled, decoupled_v1, decoupled_v2",
    )

    args = parser.parse_args()
    set_seed()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = args.batchsize
    num_workers = 2

    os.makedirs(f"models", exist_ok=True)
    os.makedirs(f"logs", exist_ok=True)

    # Original Dataset
    teachertrain_ds, teachertest_ds = get_dataset(
        args.dataset, args.augment, root=f"./data/{args.dataset}"
    )
    studenttrain_ds = teachertrain_ds
    studenttest_ds = teachertest_ds

    # augmented
    if args.augment or args.bias_eval:
        if args.bias_eval == "stylized":
            teachertrain_ds, teachertest_ds = get_custom_data(
                args.datasetpath, args.augment
            )
        elif args.bias_eval == "noised":
            teachertrain_ds, teachertest_ds = get_noised_data(
                args.dataset, noise_size=100, root=f"./data/{args.dataset}"
            )
        elif args.bias_eval == "scrambled":
            teachertrain_ds, teachertest_ds = get_scrambled_data(
                args.dataset, patch_size=56, root=f"./data/{args.dataset}"
            )
        else:
            teachertrain_ds, teachertest_ds = get_dataset(
                args.dataset, args.augment, root=f"./data/{args.dataset}"
            )

    # original
    studenttrain_loader = get_dataloader(
        studenttrain_ds, batch_size, is_train=True, num_workers=num_workers
    )
    studenttest_loader = get_dataloader(
        studenttest_ds, batch_size, is_train=False, num_workers=num_workers
    )

    # augmented/original
    teachertrain_loader = get_dataloader(
        teachertrain_ds, batch_size, is_train=True, num_workers=num_workers
    )
    teachertest_loader = get_dataloader(
        teachertest_ds, batch_size, is_train=False, num_workers=num_workers
    )

    if args.dataset == "SVHN":
        num_classes = 10
    else:
        num_classes = len(studenttrain_ds.classes)

    teachermodel = TeacherModel(args.teachermodel, num_classes)
    studentmodel = StudentModel(args.studentmodel, num_classes)

    print("============================================")
    print(f"Using device: {device}")
    print(
        f"Teacher Model: {args.teachermodel} with parameters {count_parameters(teachermodel)}"
    )
    print(
        f"Student Model {args.studentmodel} with parameters {count_parameters(studentmodel)}"
    )
    print(f"Finetuning Epochs {args.epochs[0]}")
    print(f"Distillation Epochs {args.epochs[1]}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batchsize}")
    if args.augment:
        print("Running with Augmented Data")
    else:
        print("Running without Augmented Data")
    print("============================================")
    if args.bias_eval:
        print(f"Running Bias Evaluation on {args.bias_eval} data")
    print("============================================")
    teacheroptimizer = torch.optim.AdamW(
        teachermodel.parameters(), lr=args.learningrates[0]
    )
    studentoptimizer = torch.optim.AdamW(
        studentmodel.parameters(), lr=args.learningrates[1]
    )

    criterion = nn.CrossEntropyLoss()

    if args.teachermodel_path:
        print("Loading in the teacher model path...")
        teachermodel.load_state_dict(torch.load(args.teachermodel_path))
        print("Teacher Model Loaded...")
    else:
        print("Finetuning teacher model...")
        teacher = Finetune(
            teachermodel,
            teachertrain_loader,
            teachertest_loader,
            teacheroptimizer,
            criterion,
            device,
            args.epochs[0],
            args.teacher_dir,
        )
        teacher_trainacc, teacher_testacc, teacher_losses = teacher.train(
            "logs", "models"
        )

    if args.studentmodel_path:
        print("Loading in the student model path...")
        studentmodel.load_state_dict(torch.load(args.studentmodel_path))
        print("Student Model Loaded...")
    else:
        print("Finetuning student model...")
        student = Finetune(
            studentmodel,
            studenttrain_loader,
            teachertest_loader,
            studentoptimizer,
            criterion,
            device,
            args.epochs[1],
            args.student_dir,
        )
        student_trainacc, student_testacc, student_losses = student.train(
            "logs", "models"
        )

    logitmatching = StudentModel(args.studentmodel, num_classes)
    decoupledkd = StudentModel(args.studentmodel, num_classes)
    decoupled_sim = StudentModel(args.studentmodel, num_classes)
    decoupled_sim2 = StudentModel(args.studentmodel, num_classes)  # cross_covariance

    logitmatchingoptimizer = torch.optim.AdamW(
        logitmatching.parameters(), lr=args.learningrates[1]
    )
    decoupledkdoptimizer = torch.optim.AdamW(
        decoupledkd.parameters(), lr=args.learningrates[1]
    )
    decoupled_sim_optimizer = torch.optim.AdamW(
        decoupled_sim.parameters(), lr=args.learningrates[1]
    )
    decoupled_sim2_optimizer = torch.optim.AdamW(
        decoupled_sim2.parameters(), lr=args.learningrates[1]
    )

    # Logit Matching
    if args.experiment == "logit_matching":
        print("Distilling knowledge using Logit Matching...")
        logit_model = KnowledgeDistillation(
            teachermodel,
            logitmatching,
            teachertrain_loader,
            teachertest_loader,
            logitmatchingoptimizer,
            device,
            args,
            type=f"logit_matching",
        )
        logit_model.train("logs", "models")

    elif args.experiment == "decoupled":
        # Decoupled Knowledge Distillation
        print("Distilling knowledge using DKD...")
        dkd_model = KnowledgeDistillation(
            teachermodel,
            decoupledkd,
            teachertrain_loader,
            teachertest_loader,
            decoupledkdoptimizer,
            device,
            args,
            type=f"decoupled_{args.dataset}",
        )
        dkd_model.train("logs", "models")

    elif args.experiment == "decoupled_v1":
        # Decoupled Knowledge Distillation with similarity
        print("Distilling knowledge using DKD with alignment")
        dkd_model = KnowledgeDistillation(
            teachermodel,
            decoupled_sim,
            teachertrain_loader,
            teachertest_loader,
            decoupled_sim_optimizer,
            device,
            args,
            type=f"decoupled_v1_{args.dataset}",
            v1=True,
        )
        dkd_model.train("logs", "models")

    elif args.experiment == "decoupled_v2":
        print("Distilling knowledge using DKD with alignment and cross covariance")
        dkd_model = KnowledgeDistillation(
            teachermodel,
            decoupled_sim2,
            teachertrain_loader,
            teachertest_loader,
            decoupled_sim2_optimizer,
            device,
            args,
            type=f"decoupled_v2_{args.dataset}",
            v2=True,
        )
        dkd_model.train("logs", "models")
    else:
        raise ValueError(
            "Invalid experiment. Choose from logit_matching, decoupled, decoupled_v1, decoupled_v2."
        )

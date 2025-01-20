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
    parser = argparse.ArgumentParser(description="Train a student model using knowledge distillation.")
    parser.add_argument(
        "--teachermodel",
        type=str,
        default="resnet18",
        help="Teacher model architecture",)
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
        default=15,
        help="Number of epochs",
    )
    parser.add_argument(
        "--hyperparameters",
        type=float,
        nargs="+",
        default=[0.5, 0.5, 8, 6, 0.3, 3],
        help="Hyperparameters for distillation loss [alpha, beta, mu, epsilon, decay, temperature]",
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
        "--learningrate",
        type=float,
        default=0.001,
        help="Learning rate",
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
    train_ds, test_ds = get_dataset(
        args.dataset, args.augment, root=f"./data/{args.dataset}"
    )
    distill_train_ds = train_ds
    distill_test_ds = test_ds
    
    # Augmented Dataset
    if args.augment or args.bias_eval:
        if args.bias_eval == "stylized":
            train_ds, test_ds = get_custom_data(
                args.datasetpath, args.augment
            )
        elif args.bias_eval == "noised":
            train_ds, test_ds = get_noised_data(
                args.dataset, noise_size=100, root=f"./data/{args.dataset}"
            )
        elif args.bias_eval == "scrambled":
            train_ds, test_ds = get_scrambled_data(
                args.dataset, patch_size=56, root=f"./data/{args.dataset}"
            )
        else:
            train_ds, test_ds = get_dataset(
                args.dataset, args.augment, root=f"./data/{args.dataset}"
            )

    # TrainLoader For student and teacher
    train_loader = get_dataloader(
        train_ds, batch_size, is_train=True, num_workers=num_workers
    )
    test_loader = get_dataloader(
        test_ds, batch_size, is_train=False, num_workers=num_workers
    )

    # TrainLoaders for Distillers
    distill_trainloader = get_dataloader(
        distill_train_ds, batch_size, is_train=True, num_workers=num_workers
    )
    distill_testloader = get_dataloader(
        distill_test_ds, batch_size, is_train=False, num_workers=num_workers
    )

    if args.dataset == "SVHN":
        num_classes = 10
    else:
        num_classes = len(train_ds.classes)

    # Initializaing Teacher and Student Model
    teachermodel = TeacherModel(args.teachermodel, num_classes)
    studentmodel = StudentModel(args.studentmodel, num_classes)


    print("============================================")
    print(f"Using device: {device}")
    print(f"Teacher Model: {args.teachermodel} with parameters {count_parameters(teachermodel)}")
    print(f"Student Model {args.studentmodel} with parameters {count_parameters(studentmodel)}")
    print(f"Number of Epochs {args.epochs}")
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


    # Optimizer and Loss Function
    teacheroptimizer = torch.optim.AdamW(
        teachermodel.parameters(), lr=args.learningrate
    )
    studentoptimizer = torch.optim.AdamW(
        studentmodel.parameters(), lr=args.learningrate
    )
    criterion = nn.CrossEntropyLoss()

    # Training/Loading Teacher Model.
    if args.teachermodel_path:
        print("Loading in the teacher model path...")
        teachermodel.load_state_dict(torch.load(args.teachermodel_path))
        print("Teacher Model Loaded...")
    elif args.experiment =="teacher":
        print("Finetuning teacher model...")
        teacher = Finetune(
            teachermodel,
            train_loader,
            test_loader,
            teacheroptimizer,
            criterion,
            device,
            args.epochs,
            args.teacher_dir,
        )
        teacher_trainacc, teacher_testacc, teacher_losses = teacher.train(
            "logs", "models"
        )

    # Training/Loading Student Model
    if args.studentmodel_path:
        print("Loading in the student model path...")
        studentmodel.load_state_dict(torch.load(args.studentmodel_path))
        print("Student Model Loaded...")
    elif args.experiment=="student":
        print("Finetuning student model...")
        student = Finetune(
            studentmodel,
            train_loader,
            test_loader,
            studentoptimizer,
            criterion,
            device,
            args.epochs,
            args.student_dir,
        )
        student_trainacc, student_testacc, student_losses = student.train(
            "logs", "models"
        )

    # Initializing Logit Matching, Decoupled KD, Decoupled KD with Alignment, Decoupled KD with Divergence
    logitmatching = StudentModel(args.studentmodel, num_classes)
    decoupledkd = StudentModel(args.studentmodel, num_classes)
    decoupled_align = StudentModel(args.studentmodel, num_classes)  
    decoupled_divergence = StudentModel(args.studentmodel, num_classes)  # cross_covariance

    # Distiller optimizers
    logitmatchingoptimizer = torch.optim.AdamW(
        logitmatching.parameters(), lr=args.learningrate
    )
    decoupledkdoptimizer = torch.optim.AdamW(
        decoupledkd.parameters(), lr=args.learningrate
    )
    decoupled_align_optimizer = torch.optim.AdamW(
        decoupled_align.parameters(), lr=args.learningrate
    )
    decoupled_divergence_optimizer = torch.optim.AdamW(
        decoupled_divergence.parameters(), lr=args.learningrate
    )


    # Distillation Process
    if args.experiment == 'logit_matching':
        print("Distilling knowledge using Logit Matching...")
        logit_model = KnowledgeDistillation(
            teachermodel,
            logitmatching,
            distill_trainloader,
            distill_testloader,
            logitmatchingoptimizer,
            device,
            args,
            type=f"logit_matching",
        )
        logit_model.train("logs", "models")
    
    elif args.experiment == 'decoupled':
        print("Distilling knowledge using DKD...")
        dkd_model = KnowledgeDistillation(
            teachermodel,
            decoupledkd,
            distill_trainloader,
            distill_testloader,
            decoupledkdoptimizer,
            device,
            args,
            type=f"decoupled_{args.dataset}",
        )
        dkd_model.train("logs", "models")
    
    elif args.experiment == 'decoupled_v1':
        print("Distilling knowledge using DKD with Alignment Loss")
        dkd_model = KnowledgeDistillation(
            teachermodel,
            decoupled_align,
            distill_trainloader,
            distill_testloader,
            decoupled_align_optimizer,
            device,
            args,
            type=f"decoupled_v1_{args.dataset}",
            v1=True,
        )
        dkd_model.train("logs", "models")
    
    elif args.experiment == 'decoupled_v2':
        print("Distilling knowledge using DKD with Divergence Loss")
        dkd_model = KnowledgeDistillation(
            teachermodel,
            decoupled_divergence,
            distill_trainloader,
            distill_testloader,
            decoupled_divergence_optimizer,
            device,
            args,
            type=f"decoupled_v2_{args.dataset}",
            v2=True,
        )
        dkd_model.train("logs", "models")
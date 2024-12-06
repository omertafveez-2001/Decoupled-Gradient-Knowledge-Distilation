from dataset import *
from distillation import *
from DKD import *
from finetune import *
from models import *
import argparse
from utils import *
import os
import warnings 
import matplotlib.pyplot as plt
import copy

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a student model using knowledge distillation.")
    parser.add_argument("--teachermodel", type=str, default="resnet18", help="Teacher model architecture")
    parser.add_argument("--studentmodel", type=str, default="resnet18", help="Student model architecture")
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="Dataset to use")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size for training")
    parser.add_argument("--teacherepochs", type=int, default=10, help="Number of epochs to finetune the teacher model")
    parser.add_argument("--studentepochs", type=int, default=10, help="Number of epochs to finetune the student model")
    parser.add_argument("--distillepochs", type=int, default=10, help="Number of epochs to distill knowledge to the student model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for distillation loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta value for distillation loss")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature value for distillation loss")
    parser.add_argument("--ce_weight", type=float, default=0.5, help="Cross-entropy loss weight")
    parser.add_argument("--student_dir", type=str, default="student", help="Directory for saving student model")
    parser.add_argument("--teacher_dir", type=str, default="teacher", help="Directory for saving teacher model")
    parser.add_argument("--distill_dir", type=str, default="output", help="Directory for distillation saving logs and models")
    parser.add_argument("--finetunelr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup rate for training")
    parser.add_argument("--distilllr", type=float, default=0.1, help="Learning rate for distillation")

    args = parser.parse_args()
    set_seed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = args.batchsize
    num_workers = 2
    
    os.makedirs(f'models', exist_ok=True)
    os.makedirs(f'logs', exist_ok=True)
    
    train_ds, test_ds = get_dataset(args.dataset)(root=f'./data/{args.dataset}')
    train_loader = get_dataloader(train_ds, batch_size, is_train=True, num_workers=num_workers)
    test_loader = get_dataloader(test_ds, batch_size, is_train=False, num_workers=num_workers)

    if args.dataset == 'SVHN':
        num_classes = 10
    else:
        num_classes = len(train_ds.classes)
    
    teachermodel = TeacherModel(args.teachermodel, num_classes)
    studentmodel = StudentModel(args.studentmodel, num_classes)

    print("============================================")
    print(f"Using device: {device}")
    print(f"Teacher Model: {args.teachermodel} with parameters {count_parameters(teachermodel)}")
    print(f"Student Model {args.studentmodel} with parameters {count_parameters(studentmodel)}")
    print(f"Teacher Finetuning Epochs {args.teacherepochs}")
    print(f"Student Finetuning Epochs {args.studentepochs}")
    print(f"Distillation Epochs {args.distillepochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batchsize}")
    print("============================================")


    teacheroptimizer = torch.optim.AdamW(teachermodel.parameters(), lr=args.finetunelr)
    studentoptimizer = torch.optim.AdamW(studentmodel.parameters(), lr=args.finetunelr)

    # finetuning stage
    criterion = nn.CrossEntropyLoss()
    
    print("Finetuning teacher model...")
    teacher = Finetune(teachermodel, train_loader, test_loader, teacheroptimizer, criterion, device, args.teacherepochs ,args.teacher_dir)
    teacher_trainacc, teacher_testacc, teacher_losses = teacher.train("logs", "models")

    print("Finetuning Student Model")
    student = Finetune(studentmodel, train_loader, test_loader, studentoptimizer, criterion, device, args.studentepochs, args.student_dir)
    student_trainacc, student_testacc, student_losses = student.train("logs", "models")

    logitmatching = StudentModel(args.studentmodel, num_classes)
    dkd = StudentModel(args.studentmodel, num_classes)
    tckd = StudentModel(args.studentmodel, num_classes)
    nckd = StudentModel(args.studentmodel, num_classes)

    logitmatching.load_state_dict(torch.load(os.path.join("models", f"student.pth")))
    dkd.load_state_dict(torch.load(os.path.join("models", f"student.pth")))
    tckd.load_state_dict(torch.load(os.path.join("models", f"student.pth")))
    nckd.load_state_dict(torch.load(os.path.join("models", f"student.pth")))


    logitmatchingoptimizer = torch.optim.AdamW(logitmatching.parameters(), lr=args.distilllr)
    dkdoptimizer = torch.optim.AdamW(dkd.parameters(), lr=args.distilllr)
    tckdoptimizer = torch.optim.AdamW(tckd.parameters(), lr=args.distilllr)
    nckdoptimizer = torch.optim.AdamW(nckd.parameters(), lr=args.distilllr)


    # distillation stage 
    print("Distilling knowledge using Logit Matching...")
    logit_model = KnowledgeDistillation(teachermodel, logitmatching, train_loader, test_loader, logitmatchingoptimizer, device,args, type="logit_matching")
    logit_model.train("logs", "models")

    print("Distilling knowledge using Target Class Loss ...")
    tckd_model = KnowledgeDistillation(teachermodel, tckd, train_loader, test_loader, tckdoptimizer, device,args, type="tckd")
    tckd_model.train("logs", "models")

    print("Distilling knowledge using non target class loss ...")
    nckd_model = KnowledgeDistillation(teachermodel, nckd, train_loader, test_loader, nckdoptimizer, device,args, type="nckd")
    nckd_model.train("logs", "models")

    print("Distilling knowledge using DKD...")
    dkd_model = KnowledgeDistillation(teachermodel, dkd, train_loader, test_loader, dkdoptimizer, device,args, type="decoupled")
    dkd_model.train("logs", "models")

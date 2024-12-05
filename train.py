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

warnings.filterwarnings("ignore")

def visualisation_1(train_acc, test_acc, losses, title):
    plt.style.use("ggplot")
    plt.figure(figsize=(8,6))
    plt.title(f"{title} - Accuracy")
    plt.plot(train_acc, color='red', marker='.', linestyle="--", label='Train Accuracy')
    plt.plot(test_acc, color='blue', marker='.', linestyle="--", label='Test Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,6))
    plt.title(f"{title} - Loss")
    plt.plot(losses, color='green', marker='.', linestyle="--")
    plt.show()


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
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup rate for training")

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


    teacheroptimizer = torch.optim.Adam(teachermodel.parameters(), lr=args.learning_rate)
    studentoptimizer = torch.optim.Adam(studentmodel.parameters(), lr=args.learning_rate)
    distillationoptimizer = torch.optim.Adam(studentmodel.parameters(), lr=args.learning_rate)

    # finetuning stage
    criterion = nn.CrossEntropyLoss()
    
    print("Finetuning teacher model...")
    teacher = Finetune(teachermodel, train_loader, test_loader, teacheroptimizer, criterion, device, args.teacherepochs ,args.teacher_dir)
    teacher_trainacc, teacher_testacc, teacher_losses = teacher.train("logs", "models")
    visualisation_1(teacher_trainacc, teacher_testacc, teacher_losses, "Teacher Model")

    print("Finetuning Student Model")
    student = Finetune(studentmodel, train_loader, test_loader, studentoptimizer, criterion, device, args.studentepochs, args.student_dir)
    student_trainacc, student_testacc, student_losses = student.train("logs", "models")
    visualisation_1(student_trainacc, student_testacc, student_losses, "Student Model")

    # distillation stage
    print("Distilling knowledge...")
    kd_model = KnowledgeDistillation(teachermodel, studentmodel, train_loader, test_loader, distillationoptimizer, device,args)
    distill_trainacc, distill_testacc, distill_losses = kd_model.train("logs", "models")
    visualisation_1(distill_trainacc, distill_testacc, distill_losses, "Distillation Model")
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
    parser.add_argument("--teachermodel", type=str, default="resnet18", help="Teacher model architecture")
    parser.add_argument("--studentmodel", type=str, default="resnet18", help="Student model architecture")
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="Dataset to use")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size for training")
    parser.add_argument("--finetuneepochs", type=int, default=10, help="Number of epochs to finetune the model")
    parser.add_argument("--distillepochs", type=int, default=10, help="Number of epochs to distill knowledge to the student model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for distillation loss")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature value for distillation loss")
    parser.add_argument("--student_dir", type=str, default="student", help="Directory for saving student model")
    parser.add_argument("--teacher_dir", type=str, default="teacher", help="Directory for saving teacher model")
    parser.add_argument("--distill_dir", type=str, default="output", help="Directory for distillation saving logs and models")
    parser.add_argument("--finetunelr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--distilllr", type=float, default=0.1, help="Learning rate for distillation")
    parser.add_argument("--augment", type=bool, default=False, help="Augment data")
    parser.add_argument("--teachermodel_path", type=str, default=None, help="Path to the finetuned model")
    parser.add_argument("--studentmodel_path", type=str, default=None, help="Path to the finetuned student")

    args = parser.parse_args()
    set_seed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = args.batchsize
    num_workers = 2
    
    os.makedirs(f'models', exist_ok=True)
    os.makedirs(f'logs', exist_ok=True)

    if args.augment:
        IMAGE_SIZE = 224
        TRAIN_TFMS = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        TEST_TFMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        IMAGE_SIZE = 224
        TRAIN_TFMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        TEST_TFMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    
    train_ds, test_ds = get_dataset(args.dataset, args.augment, root=f'./data/{args.dataset}')
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
    print(f"Finetuning Epochs {args.finetuneepochs}")
    print(f"Distillation Epochs {args.distillepochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batchsize}")
    if args.augment:
        print("Running with Augmented Data")
    else:
        print("Running without Augmented Data")
    print("============================================")


    teacheroptimizer = torch.optim.AdamW(teachermodel.parameters(), lr=args.finetunelr)
    studentoptimizer = torch.optim.AdamW(studentmodel.parameters(), lr=args.finetunelr)

    criterion = nn.CrossEntropyLoss()

    if args.teachermodel_path:
        print("Loading in the teacher model path...")
        teachermodel.load_state_dict(torch.load(args.teachermodel_path))
        print("Teacher Model Loaded...")
    else:
        print("Finetuning teacher model...")
        teacher = Finetune(teachermodel, train_loader, test_loader, teacheroptimizer, criterion, device, args.finetuneepochs ,args.teacher_dir)
        teacher_trainacc, teacher_testacc, teacher_losses = teacher.train("logs", "models")

    if args.studentmodel_path:
        print("Loading in the student model path...")
        studentmodel.load_state_dict(torch.load(args.studentmodel_path))
        print("Student Model Loaded...")
    else:
        print("Finetuning student model...")
        student = Finetune(studentmodel, train_loader, test_loader, studentoptimizer, criterion, device, args.distillepochs, args.student_dir)
        student_trainacc, student_testacc, student_losses = student.train("logs", "models")

    
    logitmatching = StudentModel(args.studentmodel, num_classes)
    decoupledkd = StudentModel(args.studentmodel, num_classes)
    decoupled_sim = StudentModel(args.studentmodel, num_classes) # only inducing similarity
    decoupled_orth = StudentModel(args.studentmodel, num_classes, orthogonal_projection=True)
    decoupled_orth_sim = StudentModel(args.studentmodel, num_classes, orthogonal_projection=True)
    decoupled_orth_sim_2 = StudentModel(args.studentmodel, num_classes, orthogonal_projection=True)
    decoupled_sim2 = StudentModel(args.studentmodel, num_classes) # only removing similarity


    logitmatchingoptimizer = torch.optim.AdamW(logitmatching.parameters(), lr=args.distilllr)
    decoupledkdoptimizer = torch.optim.AdamW(decoupledkd.parameters(), lr=args.distilllr)
    decoupled_orth_optimizer = torch.optim.AdamW(decoupled_orth.parameters(), lr=args.distilllr)
    decoupled_orth_sim_optimizer = torch.optim.AdamW(decoupled_orth_sim.parameters(), lr=args.distilllr)
    decoupled_orth_sim_2_optimizer = torch.optim.AdamW(decoupled_orth_sim_2.parameters(), lr=args.distilllr)
    decoupled_sim_optimizer = torch.optim.AdamW(decoupled_sim.parameters(), lr=args.distilllr)
    decoupled_sim2_optimizer = torch.optim.AdamW(decoupled_sim2.parameters(), lr=args.distilllr)

    # Logit Matching
    print("Distilling knowledge using Logit Matching...")
    logit_model = KnowledgeDistillation(teachermodel, logitmatching, train_loader, test_loader, logitmatchingoptimizer, device,args, type="logit_matching")
    logit_model.train("logs", "models")

    # Decoupled Knowledge Distillation
    print("Distilling knowledge using DKD...")
    dkd_model = KnowledgeDistillation(teachermodel, decoupledkd, train_loader, test_loader, decoupledkdoptimizer, device,args, type="decoupled")
    dkd_model.train("logs", "models")

    # Decoupled Knowledge Distillation with similarity
    print("Distilling knowledge using DKD with similarity...")
    dkd_model = KnowledgeDistillation(teachermodel, decoupled_sim, train_loader, test_loader, decoupled_sim_optimizer, device,args, type="decoupled_sim", induce_sim=True)
    dkd_model.train("logs", "models")

    # Decoupled Knowledge Distillation with reduction of similarity
    print("Distilling Knowledge using DKD with reducing similarity...")
    dkd_model = KnowledgeDistillation(teachermodel, decoupled_sim2, train_loader, test_loader, decoupled_sim2_optimizer, device,args, type="decoupled_red_sim",remove_sim=True)
    dkd_model.train("logs", "models")

    # Decoupled Knowledge Distillation with orthogonal projections
    print("Distilling knowledge using DKD with orthogonal projections...")
    dkd_model = KnowledgeDistillation(teachermodel, decoupled_orth, train_loader, test_loader, decoupled_orth_optimizer, device,args, type="decoupled_orthogonal")
    dkd_model.train("logs", "models")

    # Decoupled Knowledge Distillation with orthogonal projections and similarity
    print("Distilling knowledge using DKD with orthogonal projections and cosine sim...")
    dkd_model = KnowledgeDistillation(teachermodel, decoupled_orth_sim, train_loader, test_loader, decoupled_orth_sim_optimizer, device,args, type="decoupled_orth_sim", induce_sim=True)
    dkd_model.train("logs", "models")

    # Decoupled Knowledge Distillation with orthogonal projections and reduction of similarity
    print("Distilling knowledge using DKD with orthogonal projections and reducing cosine sim...")
    dkd_model = KnowledgeDistillation(teachermodel, decoupled_orth_sim_2, train_loader, test_loader, decoupled_orth_sim_2_optimizer, device,args, type="decoupled_orth_red_sim", remove_sim=True)
    dkd_model.train("logs", "models")
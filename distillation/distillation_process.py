import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from distillation.decoupled_distillation import *
import os
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, vgg11, vgg13, vgg16, vgg19
from models.resnet import ResNet


model_parameters={}
model_parameters['resnet18'] = ([64,128,256,512],[2,2,2,2],1,False)
model_parameters['resnet34'] = ([64,128,256,512],[3,4,6,3],1,False)
model_parameters['resnet50'] = ([64,128,256,512],[3,4,6,3],4,True)
model_parameters['resnet101'] = ([64,128,256,512],[3,4,23,3],4,True)
model_parameters['resnet152'] = ([64,128,256,512],[3,8,36,3],4,True)


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
        
        if model.startswith("resnet"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    

    def forward(self, x):
        return self.model(x)

class StudentModel(nn.Module):
    def __init__(self, model, num_classes, orthogonal_projection=False):
        super(StudentModel, self).__init__()

        self.orthogonal_projection = orthogonal_projection

        if model == "resnet18":
            self.model = ResNet(model_parameters[model], 3, num_classes, self.orthogonal_projection)
        elif model == "resnet34":
            self.model = ResNet(model_parameters[model], 3, num_classes, self.orthogonal_projection)
        elif model == "resnet50":
            self.model = ResNet(model_parameters[model], 3, num_classes, self.orthogonal_projection)
        elif model == "resnet101":
            self.model = ResNet(model_parameters[model], 3, num_classes, self.orthogonal_projection)
        elif model == "vgg11":
            self.model = vgg11(pretrained=False)
        elif model == "vgg13":
            self.model = vgg13(pretrained=False)
        elif model == "vgg16":
            self.model = vgg16(pretrained=False)
        elif model == "vgg19":
            self.model = vgg19(pretrained=False)
        else:
            raise ValueError("Invalid model name")
        
        if model.startswith("resnet"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class KnowledgeDistillation:
    def __init__(self, teacher, student, train_loader, test_loader, optimizer, device, cfg, type, induce_sim=False, remove_sim=False):
        """
        Initializes the KnowledgeDistillation class.

        Args:
            teacher (nn.Module): The teacher model.
            student (nn.Module): The student model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            optimizer (torch.optim.Optimizer): Optimizer for training the student model.
            loss_fn (str): Loss function type ("label_smoothing", "logit_matching", "decoupled").
            device (torch.device): Device to run the models on (e.g., 'cpu' or 'cuda').
            
        """
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = cfg.distillepochs
        self.output_dir = cfg.distill_dir
        self.type = type
        
        self.teacher.to(device)
        self.student.to(device)

        self.DKD = DKD(self.student, self.teacher, cfg, induce_sim, remove_sim)
        self.LogitMatching = LogitMatching(self.student, self.teacher, cfg)
        

    def train_kd_step(self):
        """
        Performs a single step of knowledge distillation training.

        Returns:
            tuple: Average loss and accuracy for the training step.
        """
        running_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0 
        grad_similarities = []
        tckd_grad_norms = []
        nckd_grad_norms = []
        tckd_grad_avgs = []
        nckd_grad_avgs = []
        target_norms = []
        nontarget_norms = []
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.type.startswith("decoupled"):
                logits_student, losses = self.DKD.forward_train(inputs, labels)
                loss = losses["loss_kd"]

                tckd_grad = torch.autograd.grad(losses["loss_tckd"], logits_student, retain_graph=True)[0]
                nckd_grad = torch.autograd.grad(losses["loss_nckd"], logits_student, retain_graph=True)[0] 

                similarity = F.cosine_similarity(tckd_grad.view(tckd_grad.size(0), -1), 
                                          nckd_grad.view(nckd_grad.size(0), -1)).mean().item()

                # L2 Norm
                tckd_grad_norm = tckd_grad.norm(2).item()
                nckd_grad_norm = nckd_grad.norm(2).item()

                # Average Mean Value
                tckd_grad_avg = tckd_grad.abs().mean().item()
                nckd_grad_avg = nckd_grad.abs().mean().item()

                grad_similarities.append(similarity)
                tckd_grad_norms.append(tckd_grad_norm)
                nckd_grad_norms.append(nckd_grad_norm)
                tckd_grad_avgs.append(tckd_grad_avg)
                nckd_grad_avgs.append(nckd_grad_avg)
                target_norms.append(losses["target_norm"])
                nontarget_norms.append(losses["nontarget_norm"])

            elif self.type=="logit_matching":
                logits_student, loss = self.LogitMatching.forward_train(inputs, labels)
            elif self.type=="nckd":
                logits_student, losses = self.DKD.forward_train(inputs, labels)
                loss = losses["loss_nckd"]
            elif self.type=="tckd":
                logits_student, losses = self.DKD.forward_train(inputs, labels)
                loss = losses["loss_tckd"]

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(logits_student, 1) 
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        train_accuracy = (correct_predictions / total_predictions) * 100
        running_loss = running_loss / len(self.train_loader)

        if self.type.startswith("decoupled"):
            avg_grad_similarity = sum(grad_similarities) / len(grad_similarities)
            tckd_grad_norms = sum(tckd_grad_norms) / len(tckd_grad_norms)
            nckd_grad_norms = sum(nckd_grad_norms)/ len(nckd_grad_norms)
            tckd_grad_avgs = sum(tckd_grad_avgs)/ len(tckd_grad_avgs)
            nckd_grad_avgs = sum(nckd_grad_avgs)/ len(nckd_grad_avgs)
            target_norms = sum(target_norms)/ len(target_norms)
            nontarget_norms = sum(nontarget_norms)/ len(nontarget_norms)
            
            return running_loss, train_accuracy, avg_grad_similarity, tckd_grad_norms, nckd_grad_norms, tckd_grad_avgs, nckd_grad_avgs, target_norms, nontarget_norms
        
        return running_loss, train_accuracy

    def test(self):
        """
        Evaluates the student model on the test dataset.

        Returns:
            float: Test accuracy.
        """
        self.student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.student(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total


    def train(self, log_path, model_path):
        """
        Trains the student model with knowledge distillation over multiple epochs.

        Args:
            epochs (int): Number of epochs for training.
            output_dir (str): Directory for saving the student model and logs.
        """
        log_path = os.path.join(log_path, f"{self.output_dir}_{self.type}.csv")
        model_path = os.path.join(model_path, f"{self.output_dir}_{self.type}.pth")

        self.teacher.eval()
        self.student.train()

        
        with open(log_path, 'w') as f:
            writer = csv.writer(f)

            if self.type=="decoupled":
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc", "avg_grad_similarity", "tckd_grad_norm", "nckd_grad_norm", "tckd_grad_mean", "nckd_grad_mean", "target_norm", "nontarget_norm"])
            else:
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(self.epochs), desc="KD Epochs"):
                if self.type.startswith("decoupled"):
                    train_loss, train_accuracy, avg_grad_sim, tckd_norm, nckd_norm, tckd_grad_avgs, nckd_grad_avgs, targetnorms, nontargetnorms = self.train_kd_step()
                else:
                    train_loss, train_accuracy = self.train_kd_step()
                test_accuracy = self.test()
                
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

                if self.type.startswith("decoupled"):
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy, avg_grad_sim, tckd_norm, nckd_norm, tckd_grad_avgs, nckd_grad_avgs, targetnorms, nontargetnorms])
                else:
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])
        torch.save(self.student.state_dict(), model_path)
        print(f"Model saved to {model_path}.")
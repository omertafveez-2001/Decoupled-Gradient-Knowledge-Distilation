import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from distillation.decoupled_distillation import *
import os
import torch.nn.functional as F
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
        
        if model.startswith("resnet"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    

    def forward(self, x):
        return self.model(x)

class StudentModel(nn.Module):
    def __init__(self, model, num_classes):
        super(StudentModel, self).__init__()


        if model == "resnet18":
            self.model = resnet18(pretrained=False)
        elif model == "resnet34":
            self.model = resnet34(pretrained=False)
        elif model == "resnet50":
            self.model = resnet50(pretrained=False)
        elif model == "resnet101":
            self.model = resnet101(pretrained=False)
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
    def __init__(self, teacher, student, train_loader, test_loader, optimizer, device, cfg, type, v1=False, v2=False):
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
        self.epochs = cfg.epochs[1]
        self.output_dir = cfg.distill_dir
        self.type = type
        
        self.teacher.to(device)
        self.student.to(device)

        self.DKD = DKD(self.student, self.teacher, cfg, v1, v2)
        self.LogitMatching = LogitMatching(self.student, self.teacher, cfg)
        

    def train_kd_step(self):
        """
        Performs a single step of knowledge distillation training.

        Returns:
            tuple: Average loss and accuracy for the training step.
        """
        running_loss = 0.0
        running_covariance = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0 
        grad_similarities = []
        target_norms = []
        nontarget_norms = []
        target_grad_mags = []
        non_target_grad_mags = []
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.type.startswith("decoupled"):
                logits_student, losses = self.DKD.forward_train(inputs, labels)
                loss = losses["loss_kd"]


                grad_similarities.append(losses["gradient_simscores"])
                target_norms.append(losses["target_norm"])
                nontarget_norms.append(losses["nontarget_norm"])
                target_grad_mags.append(losses["target_grad_magnitude"])
                non_target_grad_mags.append(losses["non_target_grad_magnitude"])
                running_covariance += losses["covariance"]

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
            target_norms = sum(target_norms)/ len(target_norms)
            nontarget_norms = sum(nontarget_norms)/ len(nontarget_norms)
            running_covariance = running_loss / len(self.train_loader)
            target_grad_mags = sum(target_grad_mags) / len(target_grad_mags)
            non_target_grad_mags = sum(non_target_grad_mags) / len(non_target_grad_mags)
            
            return running_loss, train_accuracy, avg_grad_similarity, target_norms, nontarget_norms, target_grad_mags, non_target_grad_mags, running_covariance
        
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
            if self.type.startswith("decoupled"):
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc", "avg_grad_similarity", "target_norm", "nontarget_norm", "target_gradmag", "nontarget_gradmag", "covariance"])
            else:
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(self.epochs), desc="KD Epochs"):
                if self.type.startswith("decoupled"):
                    train_loss, train_accuracy, avg_grad_sim, targetnorms, nontargetnorms, target_gradmag, nontarget_gradmag, covariance = self.train_kd_step()
                else:
                    train_loss, train_accuracy = self.train_kd_step()
                test_accuracy = self.test()
                
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

                if self.type.startswith("decoupled"):
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy, avg_grad_sim, targetnorms, nontargetnorms, target_gradmag, nontarget_gradmag, covariance])
                else:
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])
        torch.save(self.student.state_dict(), model_path)
        print(f"Model saved to {model_path}.")
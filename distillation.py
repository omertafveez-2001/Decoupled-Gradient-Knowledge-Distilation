import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from DKD import DKD
import os
import torch.nn.functional as F

class KnowledgeDistillation:
    def __init__(self, teacher, student, train_loader, test_loader, optimizer, device, cfg, type):
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
        self.DKD = DKD(student, teacher, cfg)
        self.epochs = cfg.distillepochs
        self.output_dir = cfg.distill_dir
        self.type = type
        
        self.teacher.to(device)
        self.student.to(device)

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
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.type=="decoupled":
                logits_student, losses = self.DKD.forward_train(inputs, labels)
                loss = losses["loss_kd"]

                tckd_grad = torch.autograd.grad(losses["loss_tckd"], logits_student, retain_graph=True)[0]
                nckd_grad = torch.autograd.grad(losses["loss_nckd"], logits_student, retain_graph=True)[0] 

                # compute similarities
                similarity = F.cosine_similarity(tckd_grad.view(tckd_grad.size(0), -1), 
                                          nckd_grad.view(nckd_grad.size(0), -1)).mean().item()
                grad_similarities.append(similarity)

            elif self.type=="logit_matching":
                logits_student, loss = self.DKD.forward_train(inputs, labels)
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

        if self.type=="decoupled":
            avg_grad_similarity = sum(grad_similarities) / len(grad_similarities)
            return running_loss, train_accuracy, avg_grad_similarity
        
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

        train_accuracies = []
        losses = []
        test_accuracies = []
        avg_grad_similarities = []
        with open(log_path, 'w') as f:
            writer = csv.writer(f)

            if self.type=="decoupled":
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc", "avg_grad_similarity"])
            else:
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(self.epochs), desc="KD Epochs"):
                train_loss, train_accuracy, avg_grad_sim = self.train_kd_step()
                test_accuracy = self.test()
                
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

                if self.type=="decoupled":
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy, avg_grad_sim])
                else:
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
                losses.append(train_loss)
                avg_grad_similarities.append(avg_grad_sim)

        torch.save(self.student.state_dict(), model_path)
        print(f"Model saved to {model_path}.")
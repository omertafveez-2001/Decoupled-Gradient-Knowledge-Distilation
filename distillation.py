import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from DKD import DKD

class KnowledgeDistillation:
    def __init__(self, teacher, student, train_loader, test_loader, optimizer, device, cfg):
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
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            logits_student, losses = DKD.forward_train(inputs, labels)
            loss = losses["loss_kd"] # we only use Decoupled Loss over here.
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(logits_student, 1) 
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        train_accuracy = (correct_predictions / total_predictions) * 100
        running_loss = running_loss / len(self.train_loader)
        
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


    def train(self, epochs, output_dir):
        """
        Trains the student model with knowledge distillation over multiple epochs.

        Args:
            epochs (int): Number of epochs for training.
            output_dir (str): Directory for saving the student model and logs.
        """
        log_path = f"log/{output_dir}.csv"
        model_path = f"models/{output_dir}.pth"

        self.teacher.eval()
        self.student.train()

        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(epochs), desc="KD Epochs"):
                train_loss, train_accuracy = self.train_kd_step()
                test_accuracy = self.test()
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

                writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])

        torch.save(self.student.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

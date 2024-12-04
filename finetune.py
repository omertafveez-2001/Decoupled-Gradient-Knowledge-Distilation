import torch
import csv
from tqdm import tqdm
import os

class Finetune:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device, epochs, output_dir):
        """
        Initializes the Finetune class.

        Args:
            model (torch.nn.Module): The model to train and evaluate.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            criterion (callable): Loss function.
            device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.epochs = epochs
        self.output_dir = output_dir

    def train_step(self):
        """
        Performs a single training step over the training dataset.

        Returns:
            tuple: Average loss and accuracy for the training step.
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        return running_loss / len(self.train_loader), 100 * accuracy

    def test(self):
        """
        Evaluates the model on the test dataset.

        Returns:
            float: Test accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def train(self, log_path, model_path):
        """
        Trains the model over a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
            output_dir (str): Directory for saving the model and logs.
        """
        log_path = os.path.join(log_path, f"{self.output_dir}.csv")
        model_path = os.path.join(model_path, f"{self.output_dir}.pth")

        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(self.epochs), desc="Epochs"):
                train_loss, train_accuracy = self.train_step()
                test_accuracy = self.test()
                
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

                writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])

        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

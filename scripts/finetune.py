import torch
import csv
from tqdm import tqdm
import os


def build_lr_scheduler(optimizer, warmup_epochs=20):
    milestones = {60, 75, 90}

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        factor = 1.0
        for m in sorted(milestones):
            if epoch >= m:
                factor *= 0.1
        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class Finetune:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device, epochs, output_dir, dataset):
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
        self.dataset = dataset
        self.scheduler = build_lr_scheduler(optimizer)

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

    def train(self):
        """
        Trains the model over a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
            output_dir (str): Directory for saving the model and logs.
        """
        test_accuracies = []
        train_accuracies = []
        losses = []
        log_dir = f"logs-results/{self.dataset}"
        checkpoint_dir = f"checkpoints/{self.dataset}"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.output_dir}.csv")
        model_path = os.path.join(checkpoint_dir, f"{self.output_dir}.pth")

        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(self.epochs), desc="Epochs"):
                train_loss, train_accuracy = self.train_step()
                self.scheduler.step()
                test_accuracy = self.test()
                
                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

                writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])
                test_accuracies.append(test_accuracy)
                train_accuracies.append(train_accuracy)
                losses.append(train_loss)

        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

        return train_accuracies, test_accuracies, losses

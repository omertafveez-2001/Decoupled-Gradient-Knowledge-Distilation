import torch
import csv
import os
from tqdm import tqdm
from distillation.dkd import DKD
from distillation.logitmatching import LogitMatching
from scripts.finetune import build_lr_scheduler


class KnowledgeDistillation:
    def __init__(
        self,
        teacher,
        student,
        train_loader,
        test_loader,
        optimizer,
        device,
        cfg,
        distillation_method,
    ):
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = cfg.epochs
        self.distillation_method = distillation_method
        self.dataset = cfg.dataset

        self.student_name = cfg.student_arch
        self.teacher_name = cfg.teacher_arch
        self.save_path = cfg.distillation_save_path

        self.scheduler = build_lr_scheduler(optimizer)
        self.DKD = DKD(self.student, self.teacher, cfg, self.distillation_method)
        self.LogitMatching = LogitMatching(self.student, self.teacher, cfg)

    def train_kd_step(self, epoch):
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

            if self.distillation_method.startswith("decoupled"):
                logits_student, losses = self.DKD.forward_train(inputs, labels, epoch)
                loss = losses["loss_kd"]

                grad_similarities.append(losses["gradient_simscores"])
                target_norms.append(losses["target_norm"])
                nontarget_norms.append(losses["nontarget_norm"])
                target_grad_mags.append(losses["target_grad_magnitude"])
                non_target_grad_mags.append(losses["non_target_grad_magnitude"])
                running_covariance += losses["covariance"].item()

            elif self.distillation_method == "logit_matching":
                logits_student, loss = self.LogitMatching.forward_train(inputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits_student, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        train_accuracy = (correct_predictions / total_predictions) * 100
        running_loss = running_loss / len(self.train_loader)

        if self.distillation_method.startswith("decoupled"):
            avg_grad_similarity = sum(grad_similarities) / len(grad_similarities)
            target_norms = sum(target_norms) / len(target_norms)
            nontarget_norms = sum(nontarget_norms) / len(nontarget_norms)
            running_covariance = running_covariance / len(self.train_loader)
            target_grad_mags = sum(target_grad_mags) / len(target_grad_mags)
            non_target_grad_mags = sum(non_target_grad_mags) / len(non_target_grad_mags)

            return (
                running_loss,
                train_accuracy,
                avg_grad_similarity,
                target_norms,
                nontarget_norms,
                target_grad_mags,
                non_target_grad_mags,
                running_covariance,
            )

        return running_loss, train_accuracy

    def test(self):
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

    def train(self):
        log_dir = f"logs-results/{self.dataset}"
        checkpoint_dir = f"checkpoints/{self.dataset}"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        stem = f"{self.distillation_method}_{self.student_name}_{self.teacher_name}_{self.dataset}"
        log_path = os.path.join(log_dir, f"{stem}.csv")
        model_path = os.path.join(checkpoint_dir, f"{stem}.pth")
        self.teacher.eval()
        self.student.train()

        with open(log_path, "w") as f:
            writer = csv.writer(f)
            if self.distillation_method.startswith("decoupled"):
                writer.writerow(
                    [
                        "epochs",
                        "train_loss",
                        "train_acc",
                        "test_acc",
                        "avg_grad_similarity",
                        "target_norm",
                        "nontarget_norm",
                        "target_gradmag",
                        "nontarget_gradmag",
                        "covariance",
                    ]
                )
            else:
                writer.writerow(["epochs", "train_loss", "train_acc", "test_acc"])

            for epoch in tqdm(range(self.epochs), desc="KD Epochs"):
                if self.distillation_method.startswith("decoupled"):
                    (
                        train_loss,
                        train_accuracy,
                        avg_grad_sim,
                        targetnorms,
                        nontargetnorms,
                        target_gradmag,
                        nontarget_gradmag,
                        covariance,
                    ) = self.train_kd_step(epoch)
                else:
                    train_loss, train_accuracy = self.train_kd_step(epoch)
                self.scheduler.step()
                test_accuracy = self.test()

                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    print(
                        f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, "
                        f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%"
                    )

                if self.distillation_method.startswith("decoupled"):
                    writer.writerow(
                        [
                            epoch + 1,
                            train_loss,
                            train_accuracy,
                            test_accuracy,
                            avg_grad_sim,
                            targetnorms,
                            nontargetnorms,
                            target_gradmag,
                            nontarget_gradmag,
                            covariance,
                        ]
                    )
                else:
                    writer.writerow([epoch + 1, train_loss, train_accuracy, test_accuracy])

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            torch.save(self.student.state_dict(), model_path)
            print(f"Model saved to {model_path}.")
        else:
            print("No distillation save path provided — model not saved.")

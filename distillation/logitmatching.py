import torch.nn as nn
import torch

class LogitMatching(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(LogitMatching, self).__init__()
        self.alpha = cfg.loss_weights[0]
        self.beta = cfg.loss_weights[1]
        self.temperature = cfg.temperature
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.epochs
    
    def forward_train(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.no_grad():
            teacher_logits = self.teacher(image)

        soft_criterion = nn.KLDivLoss(reduction='batchmean')
        hard_criterion = nn.CrossEntropyLoss()

        teacher_probs = nn.functional.softmax(teacher_logits/self.temperature, dim=-1)
        student_probs = nn.functional.log_softmax(student_logits/self.temperature, dim=-1)

        distillation_loss = soft_criterion(student_probs, teacher_probs)*(self.temperature**2)
        hard_loss = hard_criterion(student_logits, target)
        loss = self.beta * distillation_loss + self.alpha * hard_loss

        return student_logits, loss
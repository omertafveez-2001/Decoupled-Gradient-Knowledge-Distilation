import torch
import torch.nn as nn
import torch.nn.functional as F
from distillation.utils import dkd_loss

class DKD(nn.Module):
    def __init__(self, student, teacher, cfg, distillation_method):
        super(DKD, self).__init__()
        self.alpha = cfg.loss_weights[0]
        self.beta = cfg.loss_weights[1]
        self.epsilon = cfg.loss_weights[2]
        self.temperature = cfg.temperature
        self.warmup = cfg.warmup
        self.student = student
        self.teacher = teacher
        self.distillation_method = distillation_method

    def forward_train(self, image, target, epochs, **kwargs):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)

        loss_ce = F.cross_entropy(logits_student, target)
        warmup_scale = min(epochs / self.warmup, 1.0)
        decoupled_loss, tckd_loss, nckd_loss, target_norm, nontarget_norm, target_grad_mag, non_target_grad, grad_sim, covariance = dkd_loss(
            logits_student, logits_teacher, target, self.alpha,
            self.beta, self.epsilon, self.temperature, self.distillation_method
        )
        losses_dict = {
            "loss_kd": loss_ce + warmup_scale * decoupled_loss,
            "loss_tckd": tckd_loss,
            "loss_nckd": nckd_loss,
            "target_norm": target_norm,
            "nontarget_norm": nontarget_norm,
            "gradient_simscores": grad_sim,
            "covariance": covariance,
            "target_grad_magnitude": target_grad_mag,
            "non_target_grad_magnitude": non_target_grad
        }
        return logits_student, losses_dict

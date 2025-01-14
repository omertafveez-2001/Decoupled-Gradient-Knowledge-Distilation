import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, gamma, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student_tckd = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_tckd = cat_mask(pred_teacher, gt_mask, other_mask)

    # Compute Wasserstein distance for TCKD
    student_cdf_tckd = torch.cumsum(pred_student_tckd, dim=1)
    teacher_cdf_tckd = torch.cumsum(pred_teacher_tckd, dim=1)
    tckd_loss = torch.mean(torch.norm(student_cdf_tckd - teacher_cdf_tckd, p=1, dim=1))

    pred_teacher_nckd = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    pred_student_nckd = F.softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )

    # Compute Wasserstein distance for NCKD
    student_cdf_nckd = torch.cumsum(pred_student_nckd, dim=1)
    teacher_cdf_nckd = torch.cumsum(pred_teacher_nckd, dim=1)
    nckd_loss = torch.mean(torch.norm(student_cdf_nckd - teacher_cdf_nckd, p=1, dim=1))

    ce_loss = F.cross_entropy(logits_student, target)

    decoupled_loss = alpha * tckd_loss + beta * nckd_loss
    # total_loss = decoupled_loss
    total_loss = gamma * ce_loss + decoupled_loss

    return total_loss, ce_loss, decoupled_loss, tckd_loss, nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(nn.Module):
    def __init__(self, student, teacher, cfg, alignment=False, cross_covariance=False):
        super(DKD, self).__init__()
        self.alpha = cfg.hyperparameters[0]
        self.beta = cfg.hyperparameters[1]
        self.gamma = cfg.hyperparameters[2]
        self.phi = cfg.hyperparameters[3]
        self.epsilon = cfg.hyperparameters[4]
        self.delta = cfg.hyperparameters[5]
        self.temperature = cfg.hyperparameters[6]
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.epochs[1]

        self.alignment = alignment
        self.cross_covariance = cross_covariance

    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)

        total_loss, ce_loss, decoupled_loss, tckd_loss, nckd_loss = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.gamma,
            self.temperature,
        )
        losses_dict = {
            "loss_t": total_loss,
            "loss_kd": decoupled_loss,
            "loss_ce": ce_loss,
            "loss_tckd": tckd_loss,
            "loss_nckd": nckd_loss,
        }
        return logits_student, losses_dict


class LogitMatching(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(LogitMatching, self).__init__()
        self.alpha = cfg.hyperparameters[0]
        self.beta = cfg.hyperparameters[1]
        self.temperature = cfg.hyperparameters[5]
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.epochs[1]

    def forward_train(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.no_grad():
            teacher_logits = self.teacher(image)

        soft_criterion = nn.KLDivLoss(reduction="batchmean")
        hard_criterion = nn.CrossEntropyLoss()

        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        student_probs = nn.functional.log_softmax(
            student_logits / self.temperature, dim=-1
        )

        distillation_loss = soft_criterion(student_probs, teacher_probs) * (
            self.temperature**2
        )
        hard_loss = hard_criterion(student_logits, target)
        loss = self.beta * distillation_loss + self.alpha * hard_loss

        return student_logits, loss

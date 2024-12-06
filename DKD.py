import torch
import torch.nn as nn
import torch.nn.functional as F

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_student, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )

    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature **2)
        / target.shape[0]
    )

    return alpha*tckd_loss + beta * nckd_loss, tckd_loss, nckd_loss

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
    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__()
        self.ce_loss_weight = cfg.ce_weight
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.temperature = cfg.temperature
        self.warmup = cfg.warmup
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.distillepochs
    
    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        decoupled_loss, tckd_loss, nckd_loss = dkd_loss(logits_student, logits_teacher, target, self.alpha, self.beta, self.temperature)
        loss_dkd = min(self.epochs / self.warmup, 1.0) * decoupled_loss
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "loss_tckd": tckd_loss,
            "loss_nckd": nckd_loss,
        }
        return logits_student, losses_dict

class LogitMatching(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(LogitMatching, self).__init__()
        self.ce_loss_weight = cfg.ce_weight
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.temperature = cfg.temperature
        self.warmup = cfg.warmup
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.distillepochs
    
    def forward_train(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.no_grad():
            teacher_logits = self.teacher(image)
        
        soft_criterion = nn.KLDivLoss(reduction='batchmean')
        hard_criterion = nn.CrossEntropyLoss()
        
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = nn.functional.softmax(student_logits / self.temperature, dim=1)

        distillation_loss = soft_criterion(student_probs, teacher_probs)*(temperature**2)
        hard_loss = hard_criterion(student_logits, target)
        loss = self.beta * distillation_loss + self.alpha * hard_loss

        return student_logits, loss


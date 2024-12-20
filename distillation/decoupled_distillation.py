import torch
import torch.nn as nn
import torch.nn.functional as F

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, induce_sim=False, remove_sim=False):
    # target class logits norms
    target_student_norm = torch.norm(logits_student.gather(1, target.unsqueeze(1))).item()
    
    # Non-target class logits norms (excluding target class)
    non_target_mask = torch.ones_like(logits_student, dtype=torch.bool)
    non_target_mask.scatter_(1, target.unsqueeze(1), 0)
    non_target_student_norm = torch.norm(logits_student.masked_select(non_target_mask)).item()
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    
    log_pred_student = torch.log(pred_student)
    soft_criterion = nn.KLDivLoss(reduction="sum")
    
    tckd_loss = (
        soft_criterion(log_pred_student, pred_teacher)
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
        soft_criterion(log_pred_student_part2, pred_teacher_part2)
        * (temperature**2)
        / target.shape[0]
    )
    
    with torch.enable_grad():
        target_class_gradients = torch.autograd.grad(tckd_loss, logits_student, create_graph=True)[0]
        non_target_class_gradients = torch.autograd.grad(nckd_loss, logits_student, create_graph=True)[0]

    alignment_loss = F.cosine_similarity(
            target_class_gradients.view(target_class_gradients.size(0), -1),
            non_target_class_gradients.view(non_target_class_gradients.size(0), -1),
            dim=1
        ).mean()
    
    if induce_sim:
        total_loss = 0.25 * tckd_loss + 0.25 * nckd_loss - 0.5 * alignment_loss
    elif remove_sim:
        total_loss = 0.25 * tckd_loss + 0.25*nckd_loss + 0.5 * alignment_loss
    else:
        total_loss = alpha * tckd_loss + beta * nckd_loss
    return total_loss, tckd_loss, nckd_loss, target_student_norm, non_target_student_norm, alignment_loss.item()


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
    def __init__(self, student, teacher, cfg, induce_sim =False, remove_sim=False):
        super(DKD, self).__init__()
        self.alpha = cfg.alpha
        self.beta = 1-cfg.alpha
        self.temperature = cfg.temperature
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.distillepochs

        self.induce_sim = induce_sim
        self.remove_sim = remove_sim
        
    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)

        decoupled_loss, tckd_loss, nckd_loss, target_norm, nontarget_norm, grad_sim = dkd_loss(logits_student, logits_teacher, target, self.alpha, self.beta, self.temperature, self.induce_sim, self.remove_sim)
        losses_dict = {
            "loss_kd": decoupled_loss,
            "loss_tckd": tckd_loss,
            "loss_nckd": nckd_loss,
            "target_norm": target_norm,
            "nontarget_norm": nontarget_norm,
            "gradient_simscores": grad_sim
        }
        return logits_student, losses_dict

class LogitMatching(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(LogitMatching, self).__init__()
        self.alpha = cfg.alpha
        self.beta = 1-cfg.alpha
        self.temperature = cfg.temperature
        self.student = student
        self.teacher = teacher
        self.epochs = cfg.distillepochs
    
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
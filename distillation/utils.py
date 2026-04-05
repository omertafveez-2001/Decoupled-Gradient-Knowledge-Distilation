import torch
import torch.nn as nn
import torch.nn.functional as F

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

def get_loss(tckd_loss, nckd_loss, gradients_loss, alpha, beta, epsilon, distillation_method):
    if distillation_method == "decoupled_gradient_distillation":
        return alpha * tckd_loss + beta * nckd_loss - epsilon * gradients_loss
    else:
        return alpha * tckd_loss + beta * nckd_loss


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, epsilon, temperature, distillation_method):
    # target class logits norms
    target_student_norm = torch.norm(logits_student.gather(1, target.unsqueeze(1))).item()
    
    # Non-target class logits norms (excluding target class)
    non_target_mask = torch.ones_like(logits_student, dtype=torch.bool)
    non_target_mask.scatter_(1, target.unsqueeze(1), 0)
    non_target_student_norm = torch.norm(logits_student.masked_select(non_target_mask)).item()
    
    # Fetch masks for non target and target.
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # Compute the probabilities
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # Mask the probabilities and sum the probabilities.
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
        # compute Partial derivative of the loss w.r.t the logits
        target_class_gradients = torch.autograd.grad(tckd_loss, logits_student, create_graph=True)[0]
        non_target_class_gradients = torch.autograd.grad(nckd_loss, logits_student, create_graph=True)[0]

        # Compute the magnitude (L2 norm) of gradients
        target_grad_magnitude = torch.norm(target_class_gradients, p=2)
        non_target_grad_magnitude = torch.norm(non_target_class_gradients, p=2)

        # Fetch the gradient means 
        non_target_class_gradients_mean = non_target_class_gradients.mean()
        target_class_gradients_mean = target_class_gradients.mean()

        # Center the gradients
        non_target_class_gradients_centered = non_target_class_gradients - non_target_class_gradients_mean
        target_class_gradients_centered = target_class_gradients - target_class_gradients_mean
        
        # Compute the covariance using an outter product --> Shape is [batch, batch] Pre sum. Post sum, it is a scalar. So sum of all covariances across all examples.
        covariance = (target_class_gradients_centered @ non_target_class_gradients_centered.T).sum()
        target_class_self_covariance = (target_class_gradients_centered @ target_class_gradients_centered.T).sum()
        non_target_class_self_covariance = (non_target_class_gradients_centered @ non_target_class_gradients_centered.T).sum()

        # Normalize the collective covariance sum to get a covariance coefficient representative of the entire batch's covariance.
        covariance = covariance / torch.sqrt(target_class_self_covariance * non_target_class_self_covariance)

    # Fetch the cosine similarity between the gradients    
    cosine_similarity = F.cosine_similarity(
        target_class_gradients.flatten(start_dim=1),  # Flatten across classes
        non_target_class_gradients.flatten(start_dim=1),  # Flatten across classes
        dim=1  # Compute similarity across the batch dimension
    ).mean()
    
    # gradient loss between the target and non-target class gradients
    gradients_loss = F.mse_loss(target_class_gradients, non_target_class_gradients)

    total_loss = get_loss(nckd_loss, tckd_loss, gradients_loss, alpha, beta, epsilon, distillation_method)
        
    return total_loss, tckd_loss, nckd_loss, target_student_norm, non_target_student_norm, target_grad_magnitude.item(), non_target_grad_magnitude.item(), cosine_similarity.item(), covariance

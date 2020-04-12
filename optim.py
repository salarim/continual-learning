import torch
import torch.nn.functional as F


def seprated_softmax_loss(score_mean, target, task_target_set, task_id):
    curr_targets = task_target_set[task_id]
    prev_targets = []
    for i in range(task_id):
        prev_targets.extend(task_target_set[i])
    
    curr_mask = torch.zeros(target.size()).type(torch.BoolTensor).to(target.device)
    prev_mask = torch.zeros(target.size()).type(torch.BoolTensor).to(target.device)
    for t in curr_targets:
        curr_mask = curr_mask | (target == t)
    for t in prev_targets:
        prev_mask = prev_mask | (target == t)
    curr_mask = curr_mask.view(curr_mask.shape[0])
    prev_mask = prev_mask.view(prev_mask.shape[0])

    curr_target = target[curr_mask]
    curr_score = score_mean[curr_mask]
    curr_output = F.log_softmax(curr_score, dim=1)
    prev_target = target[prev_mask]
    prev_score = score_mean[prev_mask]
    prev_output = F.log_softmax(prev_score, dim=1)

    loss = F.nll_loss(curr_output, curr_target)
    if prev_target.shape[0] > 0:
        loss += F.nll_loss(prev_output, prev_target)

    return loss

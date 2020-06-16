import math
import numpy as np

import torch
import torch.nn.functional as F


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    warmup_from = 0.01
    eta_min = args.lr * (args.gamma ** 3)
    warmup_to = eta_min + (args.lr - eta_min) * (
            1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2

    if epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def seprated_softmax_loss(score_mean, target, tasks_targets, task_id):
    curr_targets = tasks_targets[task_id]
    prev_targets = []
    for i in range(task_id):
        prev_targets.extend(tasks_targets[i])
    
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


def triplet_loss(anchor_emb, pos_emb, neg_emb, target, margin=0.02):
    dist_a = F.pairwise_distance(anchor_emb, pos_emb, 2)
    dist_b = F.pairwise_distance(anchor_emb, neg_emb, 2)
    criterion = torch.nn.MarginRankingLoss(margin = margin)
    loss_triplet = criterion(dist_a, dist_b, target)
    # loss_embedd = anchor_emb.norm(2) + pos_emb.norm(2) + neg_emb.norm(2)
    loss = loss_triplet #+ 0.001 * loss_embedd
    return loss


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, device, batch_size_l, batch_size_u, temperature, sup_loss_coef=0.5):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.batch_size_l = batch_size_l
        self.batch_size_u = batch_size_u
        self.batch_size = batch_size_l + batch_size_u
        self.temperature = temperature
        self.sup_loss_coef = sup_loss_coef

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.self_sup_negative_mask = self._get_self_sup_negative_mask()
        self.self_sup_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_self_sup_negative_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size,  k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

        # l means labeled, u means unlabeled
    def forward(self, zis_l, zjs_l, zis_u, zjs_u, targets_l):
        self._update_batch_size(zis_l, zjs_l, zis_u, zjs_u, targets_l)
        self_sup_representations = torch.cat([zis_l,zis_u,zjs_l,zjs_u], dim=0)
        self_sup_similarity_matrix = self.cosine_similarity(self_sup_representations, 
                                                            self_sup_representations)

        self_sup_loss = self._self_sup_loss(self_sup_similarity_matrix)

        sup_similarity_matrix = self_sup_similarity_matrix[:self.batch_size_l, :self.batch_size_l]
        sup_loss = self._sup_loss(sup_similarity_matrix, targets_l)

        loss = self.sup_loss_coef * sup_loss + (1.0 - self.sup_loss_coef) * self_sup_loss
        return loss

    def _update_batch_size(self, zis_l, zjs_l, zis_u, zjs_u, targets_l):
        assert zis_l.shape[0] == zjs_l.shape[0]
        assert zis_u.shape[0] == zjs_u.shape[0]
        assert zis_l.shape[0] == targets_l.shape[0]

        if self.batch_size_l != zis_l.shape[0] or self.batch_size_u != zjs_u.shape[0]:
            self.batch_size_l = zis_l.shape[0]
            self.batch_size_u = zjs_u.shape[0]
            self.batch_size = self.batch_size_l + self.batch_size_u
            self.self_sup_negative_mask = self._get_self_sup_negative_mask()

    def cosine_similarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _self_sup_loss(self, similarity_matrix):
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.self_sup_negative_mask].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.self_sup_criterion(logits, labels)

        return loss / (2 * self.batch_size)

    def _sup_loss(self, similarity_matrix, targets):
        logits = similarity_matrix / self.temperature
        logits = self._drop_diagonal(logits)

        pos_targets = self._get_sup_pos_targets(targets)
        cross_ent = self.cross_entropy(logits, pos_targets)

        loss = cross_ent.sum()

        return loss / self.batch_size_l

    def _get_sup_pos_targets(self, targets):
        targets_mat = targets.repeat(targets.shape[0], 1)
        tmp = targets.unsqueeze(dim=1).repeat(1, targets.shape[0])
        off_diagonal = ~torch.eye(targets.shape[0]).type(torch.bool).to(self.device)

        pos_targets = (targets_mat == tmp) & off_diagonal
        pos_targets = pos_targets.type(torch.float32)

        pos_targets = self._drop_diagonal(pos_targets)

        return pos_targets.to(self.device)

    def _drop_diagonal(self, x):
        mask = ~torch.eye(x.shape[0]).type(torch.bool).to(self.device)
        return x[mask].view(x.shape[0], -1)

    def cross_entropy(self, output, target):
        m = torch.nn.LogSoftmax(dim=1)
        logsoftmax = -m(output)
        logsoftmax[~target.type(torch.bool)] = 0.0
        logsoftmax = logsoftmax.sum(dim=1)
        weights = target.sum(dim=1)
        return logsoftmax / (weights + 1e-8)

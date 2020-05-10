import torch
import torch.nn.functional as F

import numpy as np


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

    def __init__(self, device, batch_size_l, batch_size_u, temperature):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.batch_size_l = batch_size_l
        self.batch_size_u = batch_size_u
        self.batch_size = batch_size_l + batch_size_u
        self.temperature = temperature

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
        self_sup_representations = torch.cat([zis_l,zis_u,zjs_l,zjs_u], dim=0)
        self_sup_similarity_matrix = self.cosine_similarity(self_sup_representations, 
                                                            self_sup_representations)

        self_sup_loss = self._self_sup_loss(self_sup_similarity_matrix)

        sup_similarity_matrix = self_sup_similarity_matrix[:self.batch_size_l, :self.batch_size_l]
        sup_loss = self._sup_loss(sup_similarity_matrix, targets_l)

        return self_sup_loss + sup_loss

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
        return logsoftmax / weights

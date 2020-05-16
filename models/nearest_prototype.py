import torch


class NearestPrototype:

    def __init__(self, sigma):
        self.sigma = sigma

        self.task_class_prototypes = {}
        self.cur_task_feats = None
        self.prev_task_feats = None

    def add_features(self, task_id, prev_feats, cur_feats, targets):
        last_task = max(self.task_class_prototypes.keys(), default=-1)
        if task_id > last_task and self.cur_task_feats is not None and \
                                   self.prev_task_feats is not None:
            self._update_prototypes(task_id)

        self._add_cur_task_prototypes(task_id, cur_feats, targets)
        if prev_feats is not None and cur_feats is not None:
            self._save_feats(prev_feats, cur_feats)

    def predict(self, feats):
        last_task = max(self.task_class_prototypes.keys(), default=-1)
        if last_task == -1:
            raise ValueError('There is no class prototypes. Train the model by add_features first.')

        if self.cur_task_feats is not None and self.prev_task_feats is not None:
           self._update_prototypes(last_task)

        class_targets = torch.zeros(len(self.task_class_prototypes[last_task])).\
            to(feats.device)
        class_prototypes = None
        for i, (target, (feat, _)) in enumerate(self.task_class_prototypes[last_task].items()):
            class_targets[i] = target
            if class_prototypes is None:
                class_prototypes = feat.unsqueeze(dim=0)
            else:
                class_prototypes = torch.cat((class_prototypes, feat.unsqueeze(dim=0)), dim=0)
        dist_matrix = ((feats.unsqueeze(dim=1) - class_prototypes.unsqueeze(dim=0))**2).sum(dim=2)
        nearest_idxs = dist_matrix.argmin(dim=1)
        
        return class_targets[nearest_idxs]
        

    def _update_prototypes(self, task_id):
        prev_task_targets = list(self.task_class_prototypes[task_id-1].keys())
        curr_task_targets = list(self.task_class_prototypes[task_id].keys())
        absent_targets =  list(set(prev_task_targets) - set(curr_task_targets))

        for target in absent_targets:
            target_prev_mean, target_prev_num = self.task_class_prototypes[task_id-1][target]
            weights = (((self.prev_task_feats - target_prev_mean)**2).sum(dim=1).sqrt() \
                / (-2*self.sigma)).exp()
            drifts = self.cur_task_feats - self.prev_task_feats
            target_estimated_drift = (drifts * weights.unsqueeze(1)).sum(dim=0) / (weights.sum() + 1e-8)
            target_estimated_mean = target_prev_mean + target_estimated_drift
            self.task_class_prototypes[task_id][target] = (target_estimated_mean, target_prev_num)

        self.cur_task_feats = None
        self.prev_task_feats = None

    def _add_cur_task_prototypes(self, task_id, feats, targets):
        for target in targets.unique():
            target_feats = feats[targets == target]
            target = target.item()
            if task_id not in self.task_class_prototypes:
                self.task_class_prototypes[task_id] = {}
            if target not in self.task_class_prototypes[task_id]:
                self.task_class_prototypes[task_id][target] = (target_feats.mean(dim=0), 
                                                               target_feats.shape[0])
            else:
                sum_feats = target_feats.sum(dim=0) + \
                            self.task_class_prototypes[task_id][target][0] * \
                            self.task_class_prototypes[task_id][target][1]
                num = target_feats.shape[0] + self.task_class_prototypes[task_id][target][1]
                self.task_class_prototypes[task_id][target] = (sum_feats / num, num)

    def _save_feats(self, prev_feats, cur_feats):
        if self.prev_task_feats is None:
            self.prev_task_feats = prev_feats
        else:
            self.prev_task_feats = torch.cat((self.prev_task_feats, prev_feats), dim=0)

        if self.cur_task_feats is None:
            self.cur_task_feats = cur_feats
        else:
            self.cur_task_feats = torch.cat((self.cur_task_feats, cur_feats), dim=0)

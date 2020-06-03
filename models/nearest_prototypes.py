import torch


class NearestPrototypes:

    def __init__(self, sigma):
        self.sigma = sigma
        self.proto_size = 1 #TODO
        self.k = 1 #TODO

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

        all_proto_size = 0
        for proto_feats in self.task_class_prototypes[last_task].values():
            all_proto_size += proto_feats.shape[0]
        class_targets = torch.zeros(all_proto_size, dtype=torch.int64).to(feats.device)
        class_prototypes = None

        idx = 0
        for target, proto_feats in self.task_class_prototypes[last_task].items():
            class_targets[idx:idx+proto_feats.shape[0]] = target
            idx += proto_feats.shape[0]
            if class_prototypes is None:
                class_prototypes = proto_feats
            else:
                class_prototypes = torch.cat((class_prototypes, proto_feats), dim=0)

        dist_matrix = ((feats.unsqueeze(dim=1) - class_prototypes.unsqueeze(dim=0))**2).sum(dim=2)
        topk_idxs = torch.topk(dist_matrix, min(self.k, dist_matrix.shape[0]), largest=False).indices
        topk_targets = class_targets[topk_idxs]
        
        return torch.mode(topk_targets).values

    def _update_prototypes(self, task_id):
        prev_task_targets = list(self.task_class_prototypes[task_id-1].keys())
        curr_task_targets = list(self.task_class_prototypes[task_id].keys())
        absent_targets =  list(set(prev_task_targets) - set(curr_task_targets))

        for target in absent_targets:
            target_prev_protos = self.task_class_prototypes[task_id-1][target]
            new_task_class_prototypes = torch.zeros_like(self.task_class_prototypes[task_id-1][target])

            for i, target_prev_proto in enumerate(target_prev_protos): 
                weights = (((self.prev_task_feats - target_prev_proto)**2).sum(dim=1).sqrt() \
                    / (-2*self.sigma)).exp()
                drifts = self.cur_task_feats - self.prev_task_feats
                target_estimated_drift = (drifts * weights.unsqueeze(1)).sum(dim=0) \
                                          / (weights.sum() + 1e-8)
                target_estimated_proto = target_prev_proto + target_estimated_drift
                new_task_class_prototypes[i] = target_estimated_proto
            
            self.task_class_prototypes[task_id][target] = new_task_class_prototypes

        self.cur_task_feats = None
        self.prev_task_feats = None

    def _add_cur_task_prototypes(self, task_id, feats, targets):
        for target in targets.unique():
            target_feats = feats[targets == target]
            target = target.item()
            if task_id not in self.task_class_prototypes:
                self.task_class_prototypes[task_id] = {}
            if target not in self.task_class_prototypes[task_id]:
                self.task_class_prototypes[task_id][target] = target_feats[:self.proto_size]
            else:
                current_proto_size = self.task_class_prototypes[task_id][target].shape[0]
                if current_proto_size < self.proto_size:
                    new_protos = target_feats[:self.proto_size - current_proto_size]
                    old_protos = self.task_class_prototypes[task_id][target]
                    self.task_class_prototypes[task_id][target] = torch.cat((old_protos,
                                                                             new_protos), dim=0)

    def _save_feats(self, prev_feats, cur_feats):
        if self.prev_task_feats is None:
            self.prev_task_feats = prev_feats
        else:
            self.prev_task_feats = torch.cat((self.prev_task_feats, prev_feats), dim=0)

        if self.cur_task_feats is None:
            self.cur_task_feats = cur_feats
        else:
            self.cur_task_feats = torch.cat((self.cur_task_feats, cur_feats), dim=0)

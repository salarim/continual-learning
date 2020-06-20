import torch


class NearestStreamPrototype:

    def __init__(self, sigma, beta=0.997):
        self.sigma = sigma
        self.beta = beta

        self.class_prototypes = {}
        self.batch_num = 0

    def add_features(self, prev_feats, cur_feats, targets):
        self.batch_num += 1

        if prev_feats is not None:
            self._update_prototypes(prev_feats, cur_feats)

        self._add_cur_batch_prototypes(cur_feats, targets)

    def predict(self, feats):
        if not self.class_prototypes:
            raise ValueError('There is no class prototypes. Train the model by add_features first.')

        class_targets = torch.zeros(len(self.class_prototypes), dtype=torch.int64).to(feats.device)
        class_prototypes = None
        for i, (target, (feat, _)) in enumerate(self.class_prototypes.items()):
            class_targets[i] = target
            if class_prototypes is None:
                class_prototypes = feat.unsqueeze(dim=0)
            else:
                class_prototypes = torch.cat((class_prototypes, feat.unsqueeze(dim=0)), dim=0)
        dist_matrix = ((feats.unsqueeze(dim=1) - class_prototypes.unsqueeze(dim=0))**2).sum(dim=2)
        nearest_idxs = dist_matrix.argmin(dim=1)
        
        return class_targets[nearest_idxs]
        

    def _update_prototypes(self, prev_feats, cur_feats):
        absent_targets = [target for target, (_, batch_num) in self.class_prototypes.items() \
                          if batch_num < self.batch_num]

        for target in absent_targets:
            target_prev_proto, target_prev_batch_num = self.class_prototypes[target]

            weights = (((prev_feats - target_prev_proto)**2).sum(dim=1).sqrt() \
                / (-2*self.sigma)).exp()
            drifts = cur_feats - prev_feats
            target_estimated_drift = (drifts * weights.unsqueeze(1)).sum(dim=0) / (weights.sum() + 1e-8)
            target_estimated_proto = target_prev_proto + target_estimated_drift
            self.class_prototypes[target] = (target_estimated_proto, target_prev_batch_num)


    def _add_cur_batch_prototypes(self, feats, targets):
        for target in targets.unique():
            target_feats = feats[targets == target]
            target = target.item()
            if target not in self.class_prototypes:
                self.class_prototypes[target] = (target_feats.mean(dim=0), self.batch_num)
            else:
                proto, target_last_batch_num = self.class_prototypes[target]
                new_proto = target_feats.mean(dim=0)
                beta_i = self.beta ** (self.batch_num - target_last_batch_num)
                new_proto = beta_i * proto + (1 - beta_i) * new_proto
                self.class_prototypes[target] = (new_proto, self.batch_num)

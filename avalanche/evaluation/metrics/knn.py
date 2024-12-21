import torch
import torch.nn.functional as F
from avalanche.evaluation import Metric
from avalanche.evaluation import GenericPluginMetric

class KNNAccuracy(Metric[float]):
    def __init__(self, knn_k=20, knn_t=0.07):
        super().__init__()
        self.knn_k = knn_k
        self.knn_t = knn_t

        self.feature_bank = []
        self.labels_bank = []
        self.correct_count = 0
        self.total_count = 0

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: Features from the backbone for the current mini-batch.
        labels: Ground-truth labels for those samples.
        """
        if len(self.feature_bank) > 0:
            # Combine existing bank
            feat_bank_t = torch.cat(self.feature_bank, dim=0)  # shape [N, D]
            lab_bank_t = torch.cat(self.labels_bank, dim=0)    # shape [N]
            # run k-NN
            pred_labels = self._knn_predict(embeddings, feat_bank_t, lab_bank_t)
            # measure top-1
            correct = (pred_labels[:, 0] == labels).sum().item()
            self.correct_count += correct
            self.total_count += labels.size(0)
        else:
            pass

        self.feature_bank.append(embeddings.detach().cpu())
        self.labels_bank.append(labels.detach().cpu())

    def result(self) -> float:
        if self.total_count == 0:
            return 0.0
        return float(self.correct_count) / float(self.total_count)

    def reset(self):
        self.feature_bank = []
        self.labels_bank = []
        self.correct_count = 0
        self.total_count = 0

    def _knn_predict(self, query_feats, bank_feats, bank_labels):
        query_feats = F.normalize(query_feats, dim=1)
        bank_feats = F.normalize(bank_feats, dim=1)

        sim_matrix = torch.mm(query_feats, bank_feats.t())  # [B, N]
        sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)  # [B, K]
        sim_labels = bank_labels[sim_indices]  # [B, K]

        # apply weighting
        sim_weight = (sim_weight / self.knn_t).exp()
        bsz = query_feats.size(0)
        num_classes = bank_labels.max().item() + 1

        one_hot = torch.zeros(bsz * self.knn_k, num_classes, device=query_feats.device)
        one_hot.scatter_(1, sim_labels.view(-1, 1), 1.0)

        one_hot = one_hot.view(bsz, self.knn_k, -1)
        weighted_scores = one_hot * sim_weight.unsqueeze(dim=-1)
        pred_scores = weighted_scores.sum(dim=1)  # [B, num_classes]

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

class KNNAccuracyPluginMetric(GenericPluginMetric[float, KNNAccuracy]):
    """
    A plugin metric that calls `KNNAccuracy.update(...)` and logs the result
    at the times dictated by `reset_at` and `emit_at`.
    """
    def __init__(self, reset_at, emit_at, mode, knn_k=200, knn_t=0.1):
        super().__init__(
            KNNAccuracy(knn_k=knn_k, knn_t=knn_t),
            reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self):
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        with torch.no_grad():
            embeddings = strategy.model.backbone(strategy.mb_x)

        self._metric.update(embeddings, strategy.mb_y)

class MinibatchKNNAccuracy(KNNAccuracyPluginMetric):
    """K-NN accuracy tracked after every training minibatch."""
    def __init__(self, knn_k=200, knn_t=0.1):
        super().__init__(
            reset_at="iteration",
            emit_at="iteration",
            mode="train",
            knn_k=knn_k,
            knn_t=knn_t
        )

    def __str__(self):
        return "Top1_KNN_MB"


class EpochKNNAccuracy(KNNAccuracyPluginMetric):
    """K-NN accuracy tracked at the end of each training epoch."""
    def __init__(self, knn_k=200, knn_t=0.1):
        super().__init__(
            reset_at="epoch",
            emit_at="epoch",
            mode="train",
            knn_k=knn_k,
            knn_t=knn_t
        )

    def __str__(self):
        return "Top1_KNN_Epoch"

import torch
from torch import nn, Tensor
from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric


class LinearEvaluationAccuracy(AccuracyPluginMetric):
    def __init__(self, encoder: nn.Module, num_classes: int, device: str = "cpu"):
        """
        Custom metric for linear evaluation.

        :param encoder: Pretrained encoder (frozen).
        :param num_classes: Number of output classes for the linear classifier.
        :param device: Device to use (e.g., 'cpu' or 'cuda').
        """
        super().__init__(Accuracy(), reset_at="epoch", emit_at="epoch", mode="eval")
        self.encoder = encoder.to(device)
        self.linear_classifier = nn.Linear(encoder.output_dim, num_classes).to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.linear_classifier.parameters())

    def update(self, embeddings: Tensor, labels: Tensor) -> None:
        """
        Update the accuracy metric using embeddings and labels.

        :param embeddings: Frozen embeddings from the encoder.
        :param labels: Ground truth labels.
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        # Forward pass through linear classifier
        outputs = self.linear_classifier(embeddings)

        # Calculate loss (for training linear classifier)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update accuracy metric
        predictions = torch.argmax(outputs, dim=1)
        self._metric.update(predictions, labels)

    def result(self) -> float:
        return self._metric.result()

    def reset(self) -> None:
        super().reset()
        self.linear_classifier.reset_parameters()

    def __str__(self):
        return "LinearEval_Accuracy"

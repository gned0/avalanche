import math

import torch
from os.path import expanduser

from torch import nn
from torchvision.models import resnet18
from torchvision.models.quantization import resnet50

from avalanche.benchmarks import SplitCIFAR10
from avalanche.benchmarks.utils.self_supervised.simsiam_transform import SimSiamTransform
from avalanche.evaluation.metrics.linear_evaluation import LinearEvaluationAccuracy
from avalanche.models import BaseModel
from avalanche.models.self_supervised import SimSiam
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase
from avalanche.models.self_supervised.barlow_twins import BarlowTwins, BarlowTwinsLoader
from avalanche.training import Naive

from avalanche.evaluation.metrics import (
    loss_metrics, accuracy_metrics,
)
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, LRSchedulerPlugin
from avalanche.logging import InteractiveLogger
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.self_supervised_losses import SimSiamLoss, BarlowTwinsLoss, NTXentLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Self-supervised pre-training
    self_supervised_model = SimSiam(backbone=ModelBase(feature_dim=128, arch='resnet18', bn_splits=8))

    loader = SimSiamTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)

    # create the benchmark
    benchmark = SplitCIFAR10(
        n_experiences=1, dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
        train_transform=loader,
        eval_transform=loader,
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    learning_rate = 0.03
    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = SimSiamLoss()

    # create strategy
    strategy = SelfNaive(
        self_supervised_model,
        optimizer,
        criterion,
        train_epochs=30,
        device=device,
        plugins=[
            LRSchedulerPlugin(
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                step_granularity="epoch",
            )
        ],
        train_mb_size=256,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=1)
        strategy.eval(benchmark.test_stream[:])

    self_supervised_model.backbone.fc = nn.Linear(512, 10)
    save_path = "D:\EdgeDownloads\my_resnet18_weights.pth"
    torch.save(self_supervised_model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    """    
    # Linear evaluation

    pretrained_weights = "D:\EdgeDownloads\cifar10_best.pth"
    state_dict = torch.load(pretrained_weights, map_location=torch.device('cpu'))

    # Load backbone and weights
    backbone = resnet18(pretrained=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    backbone.load_state_dict(new_state_dict, strict=False)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.fc = nn.Linear(512, 10)
    for name, param in backbone.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init fc layer
    backbone.fc.weight.data.normal_(mean=0.0, std=0.01)
    backbone.fc.bias.data.zero_()
    print(backbone)


    # create the benchmark
    benchmark = SplitCIFAR10(
        n_experiences=1, dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )
    learning_rate = 0.03
    weight_decay = 1e-4
    momentum = 0.9
    optimizer = torch.optim.SGD(backbone.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # create strategy
    strategy = Naive(
        backbone,
        optimizer,
        criterion,
        train_epochs=100,
        device=device,
        train_mb_size=256,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])
    """

if __name__ == "__main__":
    main()

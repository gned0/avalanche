"""This example shows how to train a self-supervised model on CIFAR-100 using the Avalanche library.
Because CIFAR-100 comes with class labels, an online classifier can be trained,
disjointly from the SSL model via supervised learning, to evaluate the quality of the learned representations."""

import argparse
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname

# Add project root to the system path
sys.path.insert(0, abspath(dirname(__file__) + "/../.."))

from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.utils.self_supervised.transformations import (
    SimSiamTransformation, AsymmetricTransformation, SimCLRTransformation
)
from avalanche.models.self_supervised import BarlowTwins, SimSiam, BYOL, SimCLR
from avalanche.models.self_supervised.backbones.resnet import ResNet
from avalanche.training.plugins import (
    LRSchedulerPlugin, EvaluationPlugin, MomentumUpdatePlugin
)
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.self_supervised_losses import NTXentLoss, BarlowTwinsLoss, BYOLLoss, SimSiamLoss
from avalanche.training.lars import LARS
from avalanche.training.warmup_cosine_scheduler import LinearWarmupCosineAnnealingLR
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.evaluation.metrics import loss_metrics, ssl_accuracy_metrics
from avalanche.evaluation.metrics.learning_rate import learning_rate_metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = ResNet(feature_dim=args.feature_dim, cifar=True)
    num_classes = 100

    if args.ssl_method == "simsiam":
        model = SimSiam(backbone=backbone, num_classes=num_classes)
        transform = SimSiamTransformation((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = SimSiamLoss()
    elif args.ssl_method == "barlow":
        model = BarlowTwins(backbone=backbone, num_classes=num_classes)
        transform = AsymmetricTransformation((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = BarlowTwinsLoss(scale_loss=0.1)
    elif args.ssl_method == "simclr":
        model = SimCLR(backbone=backbone, num_classes=num_classes)
        transform = SimCLRTransformation((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 32)
        loss_fn = NTXentLoss(temperature=0.2)
    elif args.ssl_method == "byol":
        model = BYOL(backbone=backbone, num_classes=num_classes)
        transform = AsymmetricTransformation((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = BYOLLoss()
    else:
        raise ValueError(f"Unsupported SSL method: {args.ssl_method}")

    benchmark = SplitCIFAR100(
        n_experiences=args.n_experiences,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar100/",
        train_transform=transform,
        eval_transform=transform,
        seed=1234,
    )

    # Metrics and logging
    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]

    if args.log_file:
        text_logger = TextLogger(open(args.log_file, 'a'))
        loggers.append(text_logger)

    if args.tb_path:
        loggers.append(TensorboardLogger(args.tb_path))

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True),
        learning_rate_metrics(minibatch=True),
        ssl_accuracy_metrics(experience=True, stream=True, epoch=True),
        loggers=loggers,
    )

    # Optimizer and strategy
    train_mb_size = 256

    optimizer = LARS(
        params=model.parameters(), lr=0.3, weight_decay=1e-4, exclude_bias_n_norm=True, clip_lr=True, eta=0.02
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer, warmup_epochs=10, max_epochs=args.epochs, warmup_start_lr=0.003, eta_min=1e-6
    )
    # Add more plugins to the list if needed
    plugins = [LRSchedulerPlugin(
        scheduler=scheduler,
        step_granularity="epoch",
        reset_scheduler=True,
        reset_lr=True,
    )]
    # If using BYOL, add MomentumUpdatePlugin to update target network
    if model.ssl_method == "byol":
        plugins.append(MomentumUpdatePlugin())

    strategy = SelfNaive(
        model=model,
        optimizer=optimizer,
        criterion=loss_fn,
        train_epochs=args.epochs,
        device=device,
        plugins=plugins,
        train_mb_size=train_mb_size,
        evaluator=eval_plugin,
    )


    for i, experience in enumerate(benchmark.train_stream):
        print("Start training on experience ", experience.current_experience)
        print(f"Classes: {experience.classes_in_this_experience}")

        strategy.make_optimizer(reset_optimizer_state=True)
        # Make warmup scheduler work with SchedulerPlugin
        scheduler.optimizer = strategy.optimizer
        scheduler.last_epoch = -1
        scheduler.step()

        strategy.train(experience, num_workers=8, persistent_workers=True, drop_last=True)
        strategy.eval(benchmark.test_stream[:], num_workers=8)

    # Save weights
    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_method", type=str, default="barlow",
                        help="Self-supervised learning method: simsiam, barlow, or simclr.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model weights.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="Optional: Path to a checkpoint to load backbone weights. If not provided, backbone starts from scratch.")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory to save Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    args = parser.parse_args()
    main(args)

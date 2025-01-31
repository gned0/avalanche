import argparse
import math
from sched import scheduler

import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname


sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from torch.optim.lr_scheduler import CosineAnnealingLR
from avalanche.models.self_supervised import BarlowTwins
from avalanche.training.plugins.momentum_update import MomentumUpdatePlugin
from avalanche.benchmarks.utils.self_supervised.cifar_transform import CIFARTransform
from avalanche.models.self_supervised.byol import BYOL
from avalanche.evaluation.metrics.learning_rate import learning_rate_metrics
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from avalanche.benchmarks.utils.self_supervised.barlow_transform import BarlowTwinsTransform
from avalanche.benchmarks.utils.self_supervised.simclr_transform import SimCLRTransform
from avalanche.models.self_supervised.simclr import SimCLR
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.plugins import LRSchedulerPlugin, ReplayPlugin
from avalanche.training.self_supervised_losses import NTXentLoss, BarlowTwinsLoss, BYOLLoss
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = ModelBase(feature_dim=args.feature_dim, arch="resnet18", bn_splits=8)
    model = BarlowTwins(backbone=backbone, num_classes=100)

    transform = BarlowTwinsTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
    loss_fn = BarlowTwinsLoss()
    # Benchmark
    sequence = [
    11, 22, 39, 23, 42, 30, 78, 81, 64, 20, 29, 79, 15, 69, 86, 63, 55, 53, 73, 68,
    89, 67, 58, 97, 96, 92, 37, 14, 75, 51, 54, 7, 3, 6, 50, 40, 45, 4, 83, 98,
    27, 12, 8, 99, 60, 87, 28, 5, 84, 34, 82, 16, 72, 49, 59, 31, 71, 35, 66, 76,
    61, 17, 36, 62, 13, 2, 38, 94, 80, 19, 25, 18, 0, 1, 46, 74, 85, 91, 52, 77,
    21, 33, 32, 88, 93, 70, 44, 47, 26, 57, 90, 95, 48, 65, 43, 10, 9, 56, 24, 41
]

    benchmark = SplitCIFAR100(
        n_experiences=5,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar100/",
        train_transform=transform,
        eval_transform=transform,
        fixed_class_order=sequence,
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
        accuracy_metrics(minibatch=True, epoch=True, experience=True),
        learning_rate_metrics(minibatch=True),
        loggers=loggers,
    )

    # Optimizer and strategy
    sgd = torch.optim.SGD(model.parameters(), lr=0.3, weight_decay=1e-4)
    train_mb_size = 256

    n_batches_per_epoch = len(benchmark.train_stream[0].dataset) // train_mb_size
    warmup_steps = 10 * n_batches_per_epoch
    overall_steps = args.epochs * n_batches_per_epoch

    scheduler = CosineAnnealingLR(
        optimizer=sgd,
        T_max=overall_steps,
        eta_min=0.0,
    )

    strategy = SelfNaive(
        model=model,
        optimizer=sgd,
        criterion=loss_fn,
        train_epochs=args.epochs,
        device=device,
        plugins=[
            LRSchedulerPlugin(
                scheduler=scheduler,
                step_granularity="iteration",
            ),
            MomentumUpdatePlugin(),
        ],
        train_mb_size=train_mb_size,
        evaluator=eval_plugin,
    )

    # Train
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        print(f"Classes: {experience.classes_in_this_experience}")
        strategy.train(experience, num_workers=2, persistent_workers=True, drop_last=True)
        strategy.make_optimizer(reset_optimizer_state=True)
        strategy.eval(benchmark.test_stream[:], num_workers=2)

    # Save weights
    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_method", type=str, default="simsiam",
                        help="Self-supervised learning method: simsiam, barlow, or simclr.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory to save Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    args = parser.parse_args()
    main(args)

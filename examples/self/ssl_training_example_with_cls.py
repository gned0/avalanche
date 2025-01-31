import argparse
import math
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname


sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.benchmarks.utils.self_supervised.cifar_transform import CIFARTransform

from avalanche.evaluation.metrics.learning_rate import learning_rate_metrics
from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.utils.self_supervised.barlow_transform import BarlowTwinsTransform
from avalanche.benchmarks.utils.self_supervised.simclr_transform import SimCLRTransform
from avalanche.models.self_supervised.simclr import SimCLR
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.plugins import LRSchedulerPlugin, ReplayPlugin
from avalanche.training.self_supervised_losses import NTXentLoss
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = ModelBase(feature_dim=args.feature_dim, arch="resnet18", bn_splits=8)
    model = SimCLR(backbone=backbone, num_classes=100)

    transform = SimCLRTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
    loss_fn = NTXentLoss(temperature=0.1)
    # Benchmark
    benchmark = SplitCIFAR100(
        n_experiences=5,
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
        accuracy_metrics(minibatch=True, epoch=True, experience=True),
        learning_rate_metrics(minibatch=True),
        loggers=loggers,
    )

    # Optimizer and strategy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, weight_decay=5e-4, momentum=0.9)
    train_mb_size = 256
    n_batches_per_epoch = len(benchmark.train_stream[0].dataset) // train_mb_size
    n_restarts = 5

    overall_steps = args.epochs * n_batches_per_epoch
    cycle_steps = math.ceil(overall_steps * (1.0 / float(n_restarts)))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=cycle_steps,
    #     eta_min=1e-6,
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=overall_steps,
        eta_min=1e-6,
    )

    strategy = SelfNaive(
        model=model,
        optimizer=optimizer,
        criterion=loss_fn,
        train_epochs=args.epochs,
        device=device,

        train_mb_size=train_mb_size,
        evaluator=eval_plugin,
    )

    # Train
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=8, persistent_workers=True, drop_last=True)
        strategy.eval(benchmark.test_stream[:], num_workers=8)

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

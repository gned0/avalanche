"""This example shows how to train Barlow Twins with a CSSL plugin on ImageNet-100 in a class-incremental setting.
To evaluate the model, refer to the `cssl_linear_evaluation.py` script."""

import argparse
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname

from avalanche.benchmarks.utils.self_supervised import AsymmetricTransformationImageNet
from avalanche.training.plugins.lump import LUMPPlugin

sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.benchmarks import SplitCIFAR100, SplitImageNet
from avalanche.benchmarks.utils.self_supervised.transformations import (
    SimSiamTransformation, AsymmetricTransformation, SimCLRTransformation
)
from avalanche.models.self_supervised import BarlowTwins, SimSiam, BYOL, SimCLR
from avalanche.models.self_supervised.backbones.resnet import ResNet
from avalanche.training.plugins import (
    LRSchedulerPlugin, EvaluationPlugin, MomentumUpdatePlugin, CaSSLePlugin, PFRPlugin
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

    backbone = ResNet(cifar=False)

    # Mean, std and size values for ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = 224

    model = BarlowTwins(backbone=backbone)
    transform = AsymmetricTransformationImageNet(mean, std, size) # ImageNet augmentation pipeline
    loss_fn = BarlowTwinsLoss(scale_loss=0.1)

    # ImageNet-100 has to be present in the machine (full ImageNet also works).
    benchmark = SplitImageNet(
        n_experiences=5,
        dataset_root=args.dataset_root_path,
        meta_root=args.meta_root_path,
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
        loggers=loggers,
    )

    # Optimizer and strategy
    train_mb_size = 128

    optimizer = LARS(
        params=model.parameters(), lr=0.3, weight_decay=1e-4, exclude_bias_n_norm=True, clip_lr=True, eta=0.02
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer, warmup_epochs=10, max_epochs=args.epochs, warmup_start_lr=0.003, eta_min=1e-6
    )

    plugins = [LRSchedulerPlugin(
        scheduler=scheduler,
        step_granularity="epoch",
        reset_scheduler=True,
        reset_lr=True,
    )]

    if args.cssl_plugin == "cassle":
        plugins.append(CaSSLePlugin(loss=loss_fn, output_dim=2048))
    elif args.cssl_plugin == "pfr":
        plugins.append(PFRPlugin(output_dim=backbone.feature_dim))
    elif args.cssl_plugin == "lump":
        plugins.append(LUMPPlugin(buffer_size=train_mb_size, device=device, transform=transform))
    else:
        raise ValueError(f"Unsupported CSSL plugin: {args.cssl_plugin}")


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

    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cssl_plugin", type=str, default="cassle",
                        help="CSSL approach. Choose between cassle, pfr and lump.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model weights.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory to save Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--meta_root_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
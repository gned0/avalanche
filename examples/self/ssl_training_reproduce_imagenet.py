import argparse
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname


sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.benchmarks.utils.self_supervised.transformations import AsymmetricTransformationImageNet, SimSiamTransformation, \
    CIFARTransformation, SimCLRTransformation
from avalanche.models.self_supervised.backbones.resnet import ResNet
from avalanche.models.self_supervised import BarlowTwins, SimSiam
from avalanche.training.plugins.cassle import CaSSLePlugin
from avalanche.training.plugins.momentum_update import MomentumUpdatePlugin
from avalanche.models.self_supervised.byol import BYOL
from avalanche.evaluation.metrics.learning_rate import learning_rate_metrics
from avalanche.benchmarks import SplitImageNet
from avalanche.models.self_supervised.simclr import SimCLR
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.plugins import LRSchedulerPlugin, ReplayPlugin, PFRPlugin
from avalanche.training.self_supervised_losses import NTXentLoss, BarlowTwinsLoss, BYOLLoss, SimSiamLoss, \
    ContrastiveDistillLoss
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, ssl_accuracy_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.lars import LARS
from avalanche.training.warmup_cosine_scheduler import LinearWarmupCosineAnnealingLR


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = ResNet(cifar=False)

    if args.backbone_checkpoint:
        print(f"Loading weights from {args.backbone_checkpoint}")
        checkpoint = torch.load(args.backbone_checkpoint, map_location=device)
        missing_keys, unexpected_keys = backbone.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    model = BarlowTwins(backbone=backbone)

    transform = AsymmetricTransformationImageNet((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224)
    loss_fn = BarlowTwinsLoss(scale_loss=0.1)
    # Benchmark
    sequence = [
        11, 22, 39, 23, 42, 30, 78, 81, 64, 20, 29, 79, 15, 69, 86, 63, 55, 53, 73, 68,
        89, 67, 58, 97, 96, 92, 37, 14, 75, 51, 54, 7, 3, 6, 50, 40, 45, 4, 83, 98,
        27, 12, 8, 99, 60, 87, 28, 5, 84, 34, 82, 16, 72, 49, 59, 31, 71, 35, 66, 76,
        61, 17, 36, 62, 13, 2, 38, 94, 80, 19, 25, 18, 0, 1, 46, 74, 85, 91, 52, 77,
        21, 33, 32, 88, 93, 70, 44, 47, 26, 57, 90, 95, 48, 65, 43, 10, 9, 56, 24, 41
        ]

    benchmark = SplitImageNet(
        n_experiences=5,
        dataset_root=expanduser("~") + "/datasets/imagenet100/",
        train_transform=transform,
        eval_transform=transform,
        meta_root="/datasets/imagenet",
        fixed_class_order=sequence
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
        params=model.parameters(), lr=0.4, weight_decay=1e-4, exclude_bias_n_norm=True, clip_lr=True, eta=0.02
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer, warmup_epochs=10, max_epochs=args.epochs, warmup_start_lr=0.003, eta_min=1e-6
    )


    strategy = SelfNaive(
        model=model,
        optimizer=optimizer,
        criterion=loss_fn,
        train_epochs=args.epochs,
        device=device,
        plugins=[
            LRSchedulerPlugin(
                scheduler=scheduler,
                step_granularity="epoch",
                reset_scheduler=True,
                reset_lr=True,
            ),
            CaSSLePlugin(loss=loss_fn, output_dim=2048)
        ],
        train_mb_size=train_mb_size,
        evaluator=eval_plugin,
    )

    # if backbone from task 0 is provided, skip the first experience (saves times)
    skip_first_experience = args.backbone_checkpoint is not None

    for i, experience in enumerate(benchmark.train_stream):
        # if starting from exp. 0 checkpoint, plugin callbacks from exp. 0 still have to be called
        if skip_first_experience and i == 0:
            for plugin in strategy.plugins:
                if hasattr(plugin, "after_training_exp") and hasattr(plugin, "before_training"):
                    plugin.after_training_exp(strategy)
                    plugin.before_training(strategy)
            print(f"Skipping first experience ({experience.current_experience}) as backbone checkpoint is provided.")
            continue  # Skip the first experience

        print("Start training on experience ", experience.current_experience)
        print(f"Classes: {experience.classes_in_this_experience}")

        strategy.make_optimizer(reset_optimizer_state=True)

        scheduler.optimizer = strategy.optimizer
        scheduler.last_epoch = -1
        scheduler.step()

        strategy.train(experience, num_workers=16, persistent_workers=True, drop_last=True)
        # strategy.eval(benchmark.test_stream[:], num_workers=8)


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

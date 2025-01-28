import argparse
import math
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname
import os

sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.evaluation.metrics.learning_rate import learning_rate_metrics
from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.utils.self_supervised.barlow_transform import BarlowTwinsTransform
from avalanche.benchmarks.utils.self_supervised.simclr_transform import SimCLRTransform
from avalanche.benchmarks.utils.self_supervised.simsiam_transform import SimSiamTransform
from avalanche.benchmarks.utils.self_supervised.cifar_transform import CIFARTransform
from avalanche.models.self_supervised import SimSiam, SimCLR, BarlowTwins
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.self_supervised_losses import SimSiamLoss, BarlowTwinsLoss, NTXentLoss, ContrastiveDistillLoss
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.cassle import CaSSLePlugin

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    backbone = ModelBase(feature_dim=args.feature_dim, arch="resnet18", bn_splits=8)

    # Load pretrained weights if specified
    if args.backbone_weights:
        if not os.path.exists(args.backbone_weights):
            raise FileNotFoundError(f"Backbone weights file not found: {args.backbone_weights}")
        backbone.load_state_dict(torch.load(args.backbone_weights, map_location=device))
        print(f"Loaded pretrained weights from {args.backbone_weights}")

    # Select SSL method
    if args.ssl_method == "simsiam":
        model = SimSiam(backbone=backbone)
        transform = CIFARTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = SimSiamLoss()
    elif args.ssl_method == "barlow":
        model = BarlowTwins(backbone=backbone)
        transform = CIFARTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = BarlowTwinsLoss(scale_loss=0.025)
    elif args.ssl_method == "simclr":
        model = SimCLR(backbone=backbone)
        transform = SimCLRTransform((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 32)
        loss_fn = NTXentLoss(temperature=0.2)
    else:
        raise ValueError(f"Unsupported SSL method: {args.ssl_method}")

    # Benchmark
    benchmark = SplitCIFAR100(
        n_experiences=5,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar100/",
        train_transform=transform,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, weight_decay=5e-4, momentum=0.9)
    train_mb_size = 256
    n_batches_per_epoch = len(benchmark.train_stream[0].dataset) // train_mb_size
    n_restarts = 5

    overall_steps = args.epochs * n_batches_per_epoch 
    cycle_steps = math.ceil(overall_steps * (1.0 / float(n_restarts)))
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cycle_steps,
        eta_min=1e-6,
    )
    """
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
        plugins=[
            LRSchedulerPlugin(
                scheduler=scheduler,
                step_granularity="iteration",
            ),
        ],
        train_mb_size=train_mb_size,
        evaluator=eval_plugin,
    )
    
    # Train
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=8, persistent_workers=True, drop_last=True)

    # Save weights
    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    
    """
    # Train on the first experience only
    first_experience = benchmark.train_stream[0]
    print("Start training on experience ", first_experience.current_experience)
    print("Classes in the experience: ", first_experience.classes_in_this_experience)
 
    strategy.train(first_experience, num_workers=8, persistent_workers=True)

    # Save weights
    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")	
    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_method", type=str, default="simsiam",
                        help="Self-supervised learning method: simsiam, barlow, or simclr.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model weights.")
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory to save Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help="Path to pretrained weights for the backbone. If not specified, the backbone is initialized randomly.")
    args = parser.parse_args()
    main(args)

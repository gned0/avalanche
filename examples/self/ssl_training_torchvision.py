import argparse
import math
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname

sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from torchvision import models
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from avalanche.benchmarks.utils.self_supervised.simclr_transform import SimCLRTransform
from avalanche.benchmarks.utils.self_supervised.simsiam_transform import SimSiamTransform
from avalanche.benchmarks.utils.self_supervised.cifar_transform import CIFARTransform
from avalanche.models.self_supervised import SimSiam, SimCLR, BarlowTwins
from avalanche.training.plugins.cassle import CaSSLePlugin
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.self_supervised_losses import SimSiamLoss, BarlowTwinsLoss, NTXentLoss
from avalanche.evaluation.metrics import loss_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def create_resnet18():
        model = models.resnet18()
        model.fc = torch.nn.Identity()
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        return model

    backbone = create_resnet18()

    if args.backbone_checkpoint:
        print(f"Loading weights from {args.backbone_checkpoint}")
        checkpoint = torch.load(args.backbone_checkpoint, map_location=device)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        prefix = "encoder."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        missing_keys, unexpected_keys = backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        print("No weights file provided. Proceeding with the loaded (or random) weights.")

    if args.ssl_method == "simsiam":
        model = SimSiam(backbone=backbone)
        transform = SimSiamTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = SimSiamLoss()
    elif args.ssl_method == "barlow":
        model = BarlowTwins(backbone=backbone)
        transform = SimCLRTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = BarlowTwinsLoss()
    elif args.ssl_method == "simclr":
        model = SimCLR(backbone=backbone)
        transform = SimCLRTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = NTXentLoss(temperature=0.1)
    else:
        raise ValueError(f"Unsupported SSL method: {args.ssl_method}")

    benchmark = SplitCIFAR100(
        n_experiences=5,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar100/",
        train_transform=transform,
        eval_transform=transform,
        seed=1234,
    )

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]

    if args.log_file:
        text_logger = TextLogger(open(args.log_file, 'a'))
        loggers.append(text_logger)

    if args.tb_path:
        loggers.append(TensorboardLogger(args.tb_path))

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True),
        confusion_matrix_metrics(num_classes=10, save_image=False, stream=True),
        loggers=loggers,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, weight_decay=5e-4, momentum=0.9)
    train_mb_size = 256
    overall_steps = len(benchmark.train_stream[0].dataset) // train_mb_size * args.epochs

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

    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=2, persistent_workers=True, drop_last=True)

    # Save weights
    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


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
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="Optional: Path to a checkpoint to load backbone weights. If not provided, backbone starts from scratch.")
    args = parser.parse_args()
    main(args)

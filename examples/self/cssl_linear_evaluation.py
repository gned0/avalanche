"""This example shows how to perform linear evaluation via supervised classification on top of the representations
learned by a self-supervised model. One should first train a self-supervised model using the `cssl_training.py`
script. This script will load the pre-trained model and train a linear classifier on top of it"""

import sys
import argparse
import torch
from os.path import expanduser, abspath, dirname


sys.path.insert(0, abspath(dirname(__file__) + "/../.."))

from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.logging.text_logging import TextLogger
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10, SplitImageNet
from avalanche.models.self_supervised.backbones.resnet import ResNet, ProbeResNet
from avalanche.training import Naive
from avalanche.evaluation.metrics import (
    loss_metrics, accuracy_metrics, confusion_matrix_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch import nn

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading weights from {args.weights_path}")
    checkpoint = torch.load(args.weights_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    prefix = "model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    backbone = ResNet(cifar=args.cifar).to(device)

    missing_keys, unexpected_keys = backbone.model.load_state_dict(new_state_dict, strict=False)
    backbone.eval()
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    print("Backbone model loaded successfully")

    backbone_feature_dim = 512 if backbone.architecture == "resnet18" else 2048
    model = ProbeResNet(backbone, backbone_feature_dim=backbone_feature_dim, num_classes=args.num_classes).to(device)
    model.eval()

    print('Linear evaluation model:')
    print(model)
    initial_params = {name: param.clone().cpu() for name, param in model.named_parameters()}
    initial_buffers = {name: buffer.clone().cpu() for name, buffer in model.named_buffers()}
    initial_params_backbone = {name: param.clone().cpu() for name, param in model.backbone.model.named_parameters()}
    initial_buffers_backbone = {name: buffer.clone().cpu() for name, buffer in model.backbone.model.named_buffers()}

    benchmark = SplitImageNet(
        n_experiences=1,
        dataset_root=args.dataset_root_path,
        meta_root=args.meta_root_path,
    )

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]

    if args.log_file:
        text_logger = TextLogger(open(args.log_file, 'a'))
        loggers.append(text_logger)

    if args.tb_path:
        loggers.append(TensorboardLogger(args.tb_path))

    eval_plugin = EvaluationPlugin(
        loss_metrics(experience=True, stream=True, epoch=True),
        accuracy_metrics(experience=True, stream=True, epoch=True),
        confusion_matrix_metrics(num_classes=args.num_classes, save_image=False, stream=True),
        loggers=loggers,
    )

    optimizer = torch.optim.AdamW(
        model.linear.parameters(), lr=0.001
    )

    criterion = nn.CrossEntropyLoss()
    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=512,
        evaluator=eval_plugin,
    )

    for experience in benchmark.train_stream:
        print("Start training on experience", experience.current_experience)
        strategy.train(experience, num_workers=2, persistent_workers=True, drop_last=True)
        strategy.eval(benchmark.test_stream[:], num_workers=2)

    final_params = {name: param.cpu() for name, param in model.named_parameters()}
    final_buffers = {name: buffer.cpu() for name, buffer in model.named_buffers()}
    final_params_backbone = {name: param.cpu() for name, param in model.backbone.model.named_parameters()}
    final_buffers_backbone = {name: buffer.cpu() for name, buffer in model.backbone.model.named_buffers()}

    for name in initial_params:
        if not torch.equal(initial_params[name], final_params[name]):
            print(f"Parameter {name} was changed during training.")
    for name in initial_buffers:
        if not torch.equal(initial_buffers[name], final_buffers[name]):
            print(f"Buffer {name} was changed during training.")
    for name in initial_params_backbone:
        if not torch.equal(initial_params_backbone[name], final_params_backbone[name]):
            print(f"Backbone parameter {name} was changed during training (shouldn't happen).")
    for name in initial_buffers_backbone:
        if not torch.equal(initial_buffers_backbone[name], final_buffers_backbone[name]):
            print(f"Backbone buffer {name} was changed during training (shouldn't happen).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the trained model weights for the backbone.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file for text logs. If not provided, text logging is disabled.")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory for Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    parser.add_argument("--cifar", type=str, default=False,
                        help="If set, use CIFAR-style modifications (ResNet-18) for the backbone; otherwise, use ResNet-50.")
    parser.add_argument("--feature_dim", type=int, default=512,
                        help="Feature dimension passed to the backbone.")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Number of classes for the linear classifier.")
    args = parser.parse_args()
    main(args)

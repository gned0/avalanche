from pathlib import Path
import sys
import argparse
import torch
from os.path import expanduser
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.logging.text_logging import TextLogger
from avalanche.benchmarks import SplitCIFAR100
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase, ProbeModelBase
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

    # Load model weights
    state_dict = torch.load(args.weights_path, map_location=device)
    classifier = ModelBase(feature_dim=100, arch="resnet18", bn_splits=8)

    # Freeze all layers except the last fc
    for name, param in classifier.named_parameters():
        if name not in ["net.9.weight", "net.9.bias"]:
            param.requires_grad = False

    new_state_dict = {k: v for k, v in state_dict.items() if "net.9" not in k}
    msg = classifier.load_state_dict(new_state_dict, strict=False)
    assert set(msg.missing_keys) == {"net.9.weight", "net.9.bias"}

    print('Model loaded successfully')
    classifier = ProbeModelBase(classifier)

    # Capture initial state of parameters
    print('Linear evaluation model:')
    print(classifier)
    initial_params = {name: param.clone() for name, param in classifier.named_parameters()}
    initial_buffers = {name: buffer.clone() for name, buffer in classifier.named_buffers()}
    initial_params_backbone = {name: param.clone() for name, param in classifier.backbone[0].named_parameters()}
    initial_buffers_backbone = {name: buffer.clone() for name, buffer in classifier.backbone[0].named_buffers()}

    # Benchmark
    benchmark = SplitCIFAR100(
        n_experiences=1,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar100/",
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
        loss_metrics(experience=True, stream=True, epoch=True),
        accuracy_metrics(experience=True, stream=True, epoch=True),
        confusion_matrix_metrics(num_classes=100, save_image=False, stream=True),
        loggers=loggers,
    )

    # Optimizer and strategy
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    strategy = Naive(
        classifier,
        optimizer,
        criterion,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=512,
        evaluator=eval_plugin,
    )

    # Train and evaluate
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=8, persistent_workers=True, drop_last=True)
        strategy.eval(benchmark.test_stream[:], num_workers=8)

    # Capture final state of parameters
    final_params = {name: param.cpu() for name, param in classifier.named_parameters()}
    final_buffers = {name: buffer.cpu() for name, buffer in classifier.named_buffers()}
    final_params_backbone = {name: param.cpu() for name, param in classifier.backbone[0].named_parameters()}
    final_buffers_backbone = {name: buffer.cpu() for name, buffer in classifier.backbone[0].named_buffers()}

    # Check which parameters were changed
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
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory to save Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    args = parser.parse_args()
    main(args)

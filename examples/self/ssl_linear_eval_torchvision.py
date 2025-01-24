import sys
import argparse
import torch
from os.path import expanduser, abspath, dirname

from torchvision import models

sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.logging.text_logging import TextLogger
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

    backbone = models.resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = nn.Identity()

    num_classes = 100
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    classifier = backbone.to(device)

    if args.weights_path is not None:
        print(f"Loading weights from {args.weights_path}")
        checkpoint = torch.load(args.weights_path, map_location=device)

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

        missing_keys, unexpected_keys = classifier.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        print("No weights file provided. Proceeding with the loaded (or random) weights.")

    print('Model for linear evaluation:')
    print(classifier)

    for name, param in classifier.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    initial_params = {name: param.clone().cpu() for name, param in classifier.named_parameters()}
    initial_buffers = {name: buffer.clone().cpu() for name, buffer in classifier.named_buffers()}

    benchmark = SplitCIFAR100(
        n_experiences=args.n_experiences,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar100/",
    )
    # If you need CIFAR10, uncomment the following lines and comment the above:
    # benchmark = SplitCIFAR10(
    #     n_experiences=1,
    #     dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
    # )

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
        confusion_matrix_metrics(num_classes=num_classes, save_image=False, stream=True),
        loggers=loggers,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()), lr=0.001
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

    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=4, persistent_workers=True, drop_last=True)
        strategy.eval(benchmark.test_stream[:], num_workers=4)

    final_params = {name: param.cpu() for name, param in classifier.named_parameters()}
    final_buffers = {name: buffer.cpu() for name, buffer in classifier.named_buffers()}

    print("\nParameters updated during training:")
    for name in initial_params:
        if not torch.equal(initial_params[name], final_params[name]):
            print(f"Parameter {name} was changed during training.")
    print("\nBuffers updated during training:")
    for name in initial_buffers:
        if not torch.equal(initial_buffers[name], final_buffers[name]):
            print(f"Buffer {name} was changed during training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Optional: Path to the model weights. If not provided, uses torchvision weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="Optional: Path to a directory to save Tensorboard logs. If not provided, Tensorboard logging is disabled.")
    parser.add_argument("--n_experiences", type=int, default=1, help="Number of experiences for the benchmark.")
    args = parser.parse_args()
    main(args)

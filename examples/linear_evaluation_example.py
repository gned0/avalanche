import torch
from os.path import expanduser
from torch import nn
from avalanche.training.self_supervised.strategy_wrappers.self_naive import SelfNaive

"""
A simple example on how to use self-supervised models.
"""

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import InteractiveLogger
from avalanche.training.self_supervised_losses import SimSiamLoss
from avalanche.models.self_supervised import SimSiam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def linear_evaluation(model, data_loader, criterion, device):
    """Evaluate the model on the given data loader and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)  # Move images to the appropriate device
            labels = labels.to(device)  # Move labels to the appropriate device

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_supervised(model, train_loader, optimizer, criterion, device, num_epochs=3):
    """Train the model using supervised learning."""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0  # Initialize running loss for the epoch
        total_batches = len(train_loader)

        print(f'Currently on training epoch {epoch + 1}/{num_epochs}...')  # Indicate the start of the epoch

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move to device

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate the loss

        # Print average loss for the epoch
        avg_loss = running_loss / total_batches
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}\n')  # Added newline for better readability


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the benchmark
    benchmark = SplitMNIST(
        n_experiences=1, dataset_root=expanduser("~") + "/.avalanche/data/mnist/"
    )

    # Choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    self_supervised_model = SimSiam()

    # Adapt input layer for MNIST
    self_supervised_model.backbone.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    print("Linear evaluation")

    # Define transformations for MNIST (normalize to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])

    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    self_supervised_model.freeze_backbone()

    evaluation_network = nn.Sequential(self_supervised_model.backbone, nn.Linear(256, 10))  # Linear layer for classification

    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model using supervised learning
    train_supervised(evaluation_network, train_loader, optimizer, criterion, device)

    # Perform linear evaluation
    train_accuracy = linear_evaluation(evaluation_network, train_loader, criterion, device)
    test_accuracy = linear_evaluation(evaluation_network, test_loader, criterion, device)

    print(f'Linear Evaluation Accuracy on Training Set before self-supervised training: {train_accuracy:.2f}%')
    print(f'Linear Evaluation Accuracy on Test Set before self-supervised training: {test_accuracy:.2f}%')

    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.01)
    criterion = SimSiamLoss()

    self_supervised_model.unfreeze_backbone()

    # Create strategy
    strategy = SelfNaive(
        self_supervised_model,
        optimizer,
        criterion,
        train_epochs=1,
        device=device,
        train_mb_size=32,
        evaluator=eval_plugin,
    )

    # Train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])

    print("Linear evaluation")

    self_supervised_model.freeze_backbone()

    evaluation_network = nn.Sequential(self_supervised_model.backbone, nn.Linear(256, 10))  # Linear layer for classifica

    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model using supervised learning
    train_supervised(evaluation_network, train_loader, optimizer, criterion, device)

    # Perform linear evaluation
    train_accuracy = linear_evaluation(evaluation_network, train_loader, criterion, device)
    test_accuracy = linear_evaluation(evaluation_network, test_loader, criterion, device)

    print(f'Linear Evaluation Accuracy on Training Set: {train_accuracy:.2f}%')
    print(f'Linear Evaluation Accuracy on Test Set: {test_accuracy:.2f}%')


if __name__ == "__main__":
    main()

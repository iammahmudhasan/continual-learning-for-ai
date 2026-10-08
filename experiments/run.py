import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from benchmarks.split_mnist import get_split_mnist
from baselines.naive import NaiveBaseline
from baselines.ewc import EWC
from baselines.replay import ExperienceReplay
from metrics.cl_metrics import ContinualLearningMetrics


class SimpleMLP(nn.Module):
    """Standard 2-layer MLP backbone for Split-MNIST (784 -> 400 -> 400 -> 10)."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Continual Learning Benchmark Runner")
    parser.add_argument("--method", type=str, default="ewc", choices=["naive", "ewc", "replay"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_reg", type=float, default=1000.0)
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Starting Continual Learning Experiment")
    print(f"Method: {args.method.upper()} | Device: {device} | Seed: {args.seed}")
    print(f"=======================================================\n")

    train_loaders, test_loaders = get_split_mnist(batch_size=args.batch_size)
    num_tasks = len(train_loaders)

    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.method == "naive":
        cl_method = NaiveBaseline(model, optimizer, criterion, device=device)
    elif args.method == "ewc":
        cl_method = EWC(model, optimizer, criterion, lambda_reg=args.lambda_reg, device=device)
    elif args.method == "replay":
        cl_method = ExperienceReplay(model, optimizer, criterion, buffer_size=args.buffer_size, device=device)
    else:
        raise ValueError(f"Unknown method {args.method}")

    metrics = ContinualLearningMetrics(num_tasks=num_tasks)

    for current_task in range(num_tasks):
        print(f"--> Training on Task {current_task + 1}/{num_tasks}...")
        cl_method.train_task(current_task, train_loaders[current_task], epochs=args.epochs)

        print(f"    Evaluating on all observed tasks so far...")
        for eval_task in range(num_tasks):
            acc = cl_method.evaluate(test_loaders[eval_task])
            metrics.record_eval(current_task, eval_task, acc)
            if eval_task <= current_task:
                print(f"    - Task {eval_task + 1} Acc: {acc * 100:.2f}%")

    print("\n=======================================================")
    print("Final Continual Learning Evaluation Results")
    print("=======================================================")
    results = metrics.compute_all()
    print(f"Final Average Accuracy (A_T) : {results['average_accuracy'] * 100:.2f}%")
    print(f"Backward Transfer (BWT)     : {results['backward_transfer'] * 100:.2f}%")
    print(f"Forgetting Measure (FM)     : {results['forgetting_measure'] * 100:.2f}%")
    print("Accuracy Matrix R:")
    print(np.round(metrics.R * 100, 2))
    print("=======================================================\n")


if __name__ == "__main__":
    main()

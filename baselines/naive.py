import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class NaiveBaseline:
    """Sequential fine-tuning without regularization or replay buffer (Lower Bound)."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str = "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_task(self, task_id: int, dataloader: DataLoader, epochs: int = 1) -> None:
        self.model.train()
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return (correct / total) if total > 0 else 0.0

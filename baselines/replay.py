import random
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ExperienceReplay:
    """Experience Replay (ER) baseline using a bounded episodic buffer with Reservoir Sampling."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        buffer_size: int = 500,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.device = device

        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.total_seen = 0

    def update_buffer(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Reservoir sampling buffer insertion."""
        for i in range(x.size(0)):
            item = (x[i].detach().cpu(), y[i].detach().cpu())
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(item)
            else:
                idx = random.randint(0, self.total_seen)
                if idx < self.buffer_size:
                    self.buffer[idx] = item
            self.total_seen += 1

    def sample_buffer(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, sample_size)
        x_buf = torch.stack([item[0] for item in batch]).to(self.device)
        y_buf = torch.stack([item[1] for item in batch]).to(self.device)
        return x_buf, y_buf

    def train_task(self, task_id: int, dataloader: DataLoader, epochs: int = 1) -> None:
        self.model.train()
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Current task loss
                out = self.model(x)
                loss = self.criterion(out, y)

                # Replay loss
                if len(self.buffer) > 0:
                    x_buf, y_buf = self.sample_buffer(x.size(0))
                    out_buf = self.model(x_buf)
                    loss += self.criterion(out_buf, y_buf)

                loss.backward()
                self.optimizer.step()

                # Update reservoir buffer
                self.update_buffer(x, y)

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

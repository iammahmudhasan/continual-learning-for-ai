from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Regularizes model parameter shifts according to diagonal empirical Fisher Information.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        lambda_reg: float = 1000.0,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lambda_reg = lambda_reg
        self.device = device

        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}

    def compute_fisher(self, task_id: int, dataloader: DataLoader, num_samples: int = 200) -> None:
        """Estimates diagonal Fisher Information Matrix on the completed task."""
        self.model.eval()
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        total_samples = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            for i in range(x.size(0)):
                self.model.zero_grad()
                out = self.model(x[i : i + 1])
                loss = self.criterion(out, y[i : i + 1])
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach().pow(2)

                total_samples += 1
                if total_samples >= num_samples:
                    break
            if total_samples >= num_samples:
                break

        for n in fisher:
            fisher[n] /= float(total_samples)

        self.fisher_matrices[task_id] = fisher
        self.optimal_params[task_id] = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty_loss(self) -> torch.Tensor:
        """Computes quadratic penalty sum_{t} (1/2) * lambda * F_t * (theta - theta^*_t)^2."""
        loss = torch.tensor(0.0, device=self.device)
        for task_id in self.fisher_matrices:
            fisher = self.fisher_matrices[task_id]
            opt = self.optimal_params[task_id]
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in fisher:
                    loss += (fisher[n] * (p - opt[n]).pow(2)).sum()
        return 0.5 * self.lambda_reg * loss

    def train_task(self, task_id: int, dataloader: DataLoader, epochs: int = 1) -> None:
        self.model.train()
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y) + self.penalty_loss()
                loss.backward()
                self.optimizer.step()

        self.compute_fisher(task_id, dataloader)

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

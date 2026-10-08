import numpy as np
from typing import Dict, Optional


class ContinualLearningMetrics:
    """Computes standard Continual Learning metrics from an accuracy matrix R.

    R[i, j] represents the test performance on task j after training on task i (0-indexed).
    """

    def __init__(self, num_tasks: int, random_baseline: Optional[np.ndarray] = None):
        self.num_tasks = num_tasks
        self.R = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        # Random initialization accuracy b_tilde per task (default 1/C or empirical test)
        self.b_tilde = random_baseline if random_baseline is not None else np.zeros(num_tasks)

    def record_eval(self, current_task: int, evaluated_task: int, accuracy: float) -> None:
        """Record accuracy R[current_task, evaluated_task]."""
        self.R[current_task, evaluated_task] = accuracy

    def average_accuracy(self, after_task: Optional[int] = None) -> float:
        """Computes Average Accuracy A_t = (1 / (t + 1)) * sum_{j=0}^t R[t, j]."""
        t = (self.num_tasks - 1) if after_task is None else after_task
        return float(np.mean(self.R[t, : t + 1]))

    def backward_transfer(self, after_task: Optional[int] = None) -> float:
        """Computes BWT = (1 / t) * sum_{j=0}^{t-1} (R[t, j] - R[j, j]).

        Negative indicates catastrophic forgetting. Positive indicates backward facilitation.
        """
        t = (self.num_tasks - 1) if after_task is None else after_task
        if t == 0:
            return 0.0
        diffs = [self.R[t, j] - self.R[j, j] for j in range(t)]
        return float(np.mean(diffs))

    def forward_transfer(self, after_task: Optional[int] = None) -> float:
        """Computes FWT = (1 / t) * sum_{j=1}^t (R[j-1, j] - b_tilde[j]).

        Measures positive inductive bias transfer to future tasks before training on them.
        """
        t = (self.num_tasks - 1) if after_task is None else after_task
        if t == 0:
            return 0.0
        diffs = [self.R[j - 1, j] - self.b_tilde[j] for j in range(1, t + 1)]
        return float(np.mean(diffs))

    def forgetting_measure(self, after_task: Optional[int] = None) -> float:
        """Computes Forgetting Measure FM = (1 / t) * sum_{j=0}^{t-1} (max_{l < t} R[l, j] - R[t, j])."""
        t = (self.num_tasks - 1) if after_task is None else after_task
        if t == 0:
            return 0.0
        forgetting = []
        for j in range(t):
            max_past = np.max(self.R[:t, j])
            forgetting.append(max_past - self.R[t, j])
        return float(np.mean(forgetting))

    def compute_all(self, after_task: Optional[int] = None) -> Dict[str, float]:
        """Compute all primary metrics simultaneously."""
        return {
            "average_accuracy": self.average_accuracy(after_task),
            "backward_transfer": self.backward_transfer(after_task),
            "forward_transfer": self.forward_transfer(after_task),
            "forgetting_measure": self.forgetting_measure(after_task),
        }

from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_split_mnist(
    root: str = "./data",
    batch_size: int = 64,
    num_tasks: int = 5,
    classes_per_task: int = 2,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Partitions MNIST into sequential binary classification tasks.

    e.g., Task 0: [0, 1], Task 1: [2, 3], Task 2: [4, 5], Task 3: [6, 7], Task 4: [8, 9].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loaders = []
    test_loaders = []

    for task_id in range(num_tasks):
        task_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

        train_indices = [i for i, target in enumerate(train_dataset.targets) if target.item() in task_classes]
        test_indices = [i for i, target in enumerate(test_dataset.targets) if target.item() in task_classes]

        train_sub = Subset(train_dataset, train_indices)
        test_sub = Subset(test_dataset, test_indices)

        train_loaders.append(DataLoader(train_sub, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_sub, batch_size=batch_size, shuffle=False))

    return train_loaders, test_loaders

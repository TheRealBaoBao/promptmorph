"""Train the progress prior from versioned MuJoCo feature datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from promptmorph.learning.features import PROGRESS_FEATURE_NAMES
from promptmorph.learning.model import PROGRESS_CLASSES, SubgoalProgressNet


def load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as dataset:
        features = np.asarray(dataset["features"], dtype=np.float32)
        labels = np.asarray(dataset["labels"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != len(PROGRESS_FEATURE_NAMES):
        raise ValueError(
            f"features must have shape [N, {len(PROGRESS_FEATURE_NAMES)}], got {features.shape}"
        )
    if labels.shape != (features.shape[0],):
        raise ValueError(f"labels must have shape ({features.shape[0]},), got {labels.shape}")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(PROGRESS_CLASSES):
        raise ValueError(f"labels must be integers in [0, {len(PROGRESS_CLASSES) - 1}]")
    return torch.from_numpy(features), torch.from_numpy(labels)


def train(
    dataset_path: Path,
    checkpoint_path: Path,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 7,
) -> float:
    torch.manual_seed(seed)
    features, labels = load_dataset(dataset_path)
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=batch_size, shuffle=True
    )
    model = SubgoalProgressNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_value = float("nan")
    for _ in range(epochs):
        for feature_batch, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "feature_names": PROGRESS_FEATURE_NAMES,
            "class_names": PROGRESS_CLASSES,
            "seed": seed,
        },
        checkpoint_path,
    )
    return loss_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/progress_net.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    final_loss = train(args.dataset, args.output, epochs=args.epochs)
    print(f"saved {args.output} (final batch loss={final_loss:.4f})")


if __name__ == "__main__":
    main()


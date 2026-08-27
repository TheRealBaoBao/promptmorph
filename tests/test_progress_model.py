from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from promptmorph.learning.features import PROGRESS_FEATURE_NAMES  # noqa: E402
from promptmorph.learning.model import PROGRESS_CLASSES, SubgoalProgressNet  # noqa: E402
from promptmorph.learning.train import load_dataset, train  # noqa: E402


def test_progress_model_and_training_contract(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(32, len(PROGRESS_FEATURE_NAMES))).astype(np.float32)
    labels = np.arange(32, dtype=np.int64) % len(PROGRESS_CLASSES)
    dataset = tmp_path / "dataset.npz"
    checkpoint = tmp_path / "model.pt"
    np.savez(dataset, features=features, labels=labels)
    final_loss = train(dataset, checkpoint, epochs=1, batch_size=16)
    assert np.isfinite(final_loss)
    assert checkpoint.exists()
    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded["format_version"] == 1

    model = SubgoalProgressNet()
    logits = model(torch.from_numpy(features[:2]))
    assert logits.shape == (2, len(PROGRESS_CLASSES))
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(2, 3))


def test_dataset_validation(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, features=np.zeros((4, 3)), labels=np.zeros(4))
    with pytest.raises(ValueError, match="features"):
        load_dataset(invalid)


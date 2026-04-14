from __future__ import annotations

import torch

from src.train import save_checkpoint


def test_save_checkpoint_writes_meta(tmp_path):
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    meta = {"num_classes": 3, "backbone": "dummy"}
    save_path = tmp_path / "ckpt.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        metrics={"accuracy": 0.5},
        save_path=str(save_path),
        metadata=meta,
    )

    checkpoint = torch.load(str(save_path), map_location="cpu")
    assert checkpoint["meta"] == meta

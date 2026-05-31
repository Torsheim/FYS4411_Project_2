"""Save and load RBM model checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM


def save_model(model: GaussianBinaryRBM, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        num_visible=model.num_visible,
        num_hidden=model.num_hidden,
        sigma=model.sigma,
        a=model.a,
        b=model.b,
        W=model.W,
    )


def load_model(path: str | Path) -> GaussianBinaryRBM:
    with np.load(path) as data:
        return GaussianBinaryRBM(
            num_visible=int(data["num_visible"]),
            num_hidden=int(data["num_hidden"]),
            sigma=float(data["sigma"]),
            a=data["a"],
            b=data["b"],
            W=data["W"],
        )

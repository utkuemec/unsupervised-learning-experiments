"""Shared helpers: reproducible splits, dataset loading, and result persistence."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import fetch_olivetti_faces  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
METRICS_PATH = ROOT / "results" / "metrics.json"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_olivetti():
    """Olivetti faces: 400 grayscale 64x64 images of 40 people (10 each)."""
    data = fetch_olivetti_faces(shuffle=True, random_state=SEED)
    return data.data, data.target


def stratified_split(X, y, test_size=0.2, val_size=0.2):
    """60/20/20 train/val/test split, stratified so every person appears in each split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def face_grid(images, titles=None, n_cols=10, size=1.4):
    images = np.asarray(images).reshape(-1, 64, 64)
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * size, n_rows * size * 1.1))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    return fig


def record(section: str, metrics: dict) -> None:
    """Merge one experiment's metrics into results/metrics.json."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    existing[section] = metrics
    METRICS_PATH.write_text(json.dumps(existing, indent=2, default=float) + "\n")
    print(f"[{section}]", json.dumps(metrics, indent=2, default=float))

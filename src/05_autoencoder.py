"""Sparse autoencoder (PyTorch) on Olivetti faces, tuned with k-fold cross-validation.

Questions answered
  * Which bottleneck size and L1 activity penalty minimise validation reconstruction error?
  * How does the tuned autoencoder compare with PCA at the *same* code size on held-out faces?
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch import nn

from common import SEED, load_olivetti, record, save_fig, set_seed, stratified_split


class SparseAutoencoder(nn.Module):
    def __init__(self, n_in: int, code: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 512), nn.ReLU(), nn.Linear(512, code), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(code, 512), nn.ReLU(), nn.Linear(512, n_in), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train(X: np.ndarray, code: int, l1: float, epochs: int, X_val: np.ndarray | None = None):
    torch.manual_seed(SEED)
    model = SparseAutoencoder(X.shape[1], code)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.tensor(X, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb in loader:
            recon, z = model(xb)
            loss = nn.functional.mse_loss(recon, xb) + l1 * z.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    val_mse = None
    if X_val is not None:
        val_mse = reconstruct_mse(model, X_val)
    return model, val_mse


@torch.no_grad()
def reconstruct(model: SparseAutoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32))[0].numpy()


def reconstruct_mse(model: SparseAutoencoder, X: np.ndarray) -> float:
    return float(np.mean((reconstruct(model, X) - X) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    set_seed()
    torch.manual_seed(SEED)

    X, y = load_olivetti()
    X_train, X_val, X_test, *_ = stratified_split(X, y)
    X_dev = np.vstack([X_train, X_val])  # CV runs on train+val; test is untouched until the end

    codes = [64, 128] if args.quick else [32, 64, 128]
    l1s = [1e-4] if args.quick else [1e-3, 1e-4, 1e-5]
    epochs = 15 if args.quick else 80
    folds = KFold(n_splits=2 if args.quick else 3, shuffle=True, random_state=SEED)

    grid = {}
    for code, l1 in itertools.product(codes, l1s):
        scores = [train(X_dev[tr], code, l1, epochs, X_dev[va])[1] for tr, va in folds.split(X_dev)]
        grid[f"code={code}, l1={l1:g}"] = round(float(np.mean(scores)), 6)
        print(f"code={code:<4} l1={l1:<7g} cv_mse={np.mean(scores):.6f}")
    best_key = min(grid, key=grid.get)
    best_code = int(best_key.split(",")[0].split("=")[1])
    best_l1 = float(best_key.split("l1=")[1])

    model, _ = train(X_dev, best_code, best_l1, epochs)
    ae_test = reconstruct_mse(model, X_test)

    pca = PCA(n_components=best_code, random_state=SEED).fit(X_dev)
    pca_rec = pca.inverse_transform(pca.transform(X_test))
    pca_test = float(np.mean((pca_rec - X_test) ** 2))

    ae_rec = reconstruct(model, X_test)
    fig, axes = plt.subplots(3, 8, figsize=(11, 4.4))
    for i in range(8):
        for r, (img, label) in enumerate([(X_test, "original"), (ae_rec, "autoencoder"), (pca_rec, "PCA")]):
            axes[r, i].imshow(img[i].reshape(64, 64), cmap="gray")
            axes[r, i].axis("off")
            if i == 0:
                axes[r, i].set_title(label, fontsize=8, loc="left")
    fig.tight_layout()
    save_fig(fig, "autoencoder_reconstructions")

    record(
        "autoencoder",
        {
            "cv_grid_mse": grid,
            "best": {"code_size": best_code, "l1": best_l1},
            "test_mse_autoencoder": round(ae_test, 6),
            "test_mse_pca_same_code_size": round(pca_test, 6),
            "epochs": epochs,
        },
    )


if __name__ == "__main__":
    main()

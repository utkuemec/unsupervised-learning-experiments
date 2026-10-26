"""Dimensionality reduction: PCA / Incremental PCA on MNIST and Kernel PCA on a Swiss roll.

Questions answered
  * How many principal components keep 95% of MNIST's variance, and what does the
    compressed reconstruction look like?
  * Which Kernel PCA kernel "unrolls" a non-linear manifold, and which kPCA settings
    work best as a feature extractor for a downstream classifier (tuned with GridSearchCV)?
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, record, save_fig, set_seed


def mnist_pca(quick: bool) -> dict:
    # liac-arff: pure-Python parser, avoids pandas/scikit-learn version coupling
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", return_X_y=True)
    X = X.astype(np.float32) / 255.0
    if quick:
        X, y = X[:10_000], y[:10_000]

    # 2-D projection for visualisation
    pca2 = PCA(n_components=2, random_state=SEED)
    X2 = pca2.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X2[::5, 0], X2[::5, 1], c=y[::5].astype(int), cmap="tab10", s=3, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="digit")
    ax.set(title="MNIST projected onto the first 2 principal components", xlabel="PC1", ylabel="PC2")
    save_fig(fig, "pca_mnist_2d")

    # Components needed for 95% variance
    full = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(full.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)

    # Incremental PCA (out-of-core friendly) compression + reconstruction
    ipca = IncrementalPCA(n_components=k95)
    for batch in np.array_split(X, 50):
        ipca.partial_fit(batch)
    X_rec = ipca.inverse_transform(ipca.transform(X))
    mse = float(np.mean((X - X_rec) ** 2))

    idx = [int(np.where(y == str(d))[0][0]) for d in range(10)]
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
    for col, i in enumerate(idx):
        axes[0, col].imshow(X[i].reshape(28, 28), cmap="gray")
        axes[1, col].imshow(X_rec[i].reshape(28, 28), cmap="gray")
        for r in (0, 1):
            axes[r, col].axis("off")
    axes[0, 0].set_title("original", fontsize=8, loc="left")
    axes[1, 0].set_title(f"{k95} comps", fontsize=8, loc="left")
    fig.tight_layout()
    save_fig(fig, "pca_mnist_reconstruction")

    return {
        "pc1_pc2_explained_variance": [round(float(v), 4) for v in pca2.explained_variance_ratio_],
        "components_for_95pct_variance": k95,
        "compression_ratio": round(784 / k95, 2),
        "reconstruction_mse": round(mse, 5),
    }


def swiss_roll_kpca(quick: bool) -> dict:
    X, t = make_swiss_roll(n_samples=1000, noise=0.0, random_state=SEED)
    Xs = StandardScaler().fit_transform(X)

    kernels = {
        "linear": KernelPCA(n_components=2, kernel="linear"),
        "rbf": KernelPCA(n_components=2, kernel="rbf", gamma=0.04),
        "sigmoid": KernelPCA(n_components=2, kernel="sigmoid", gamma=1e-3, coef0=1),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, kpca) in zip(axes, kernels.items(), strict=True):
        Z = kpca.fit_transform(Xs)
        ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="Spectral", s=6)
        ax.set_title(f"Kernel PCA — {name}")
    fig.tight_layout()
    save_fig(fig, "kpca_swiss_roll_kernels")

    # Use kPCA as a learned feature extractor and tune it for a downstream task.
    y = (t > np.median(t)).astype(int)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kpca", KernelPCA(n_components=2)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    grid = {
        "kpca__kernel": ["linear", "rbf", "sigmoid"],
        "kpca__gamma": np.linspace(0.03, 0.05, 3 if quick else 10),
    }
    search = GridSearchCV(pipe, grid, cv=3, n_jobs=-1).fit(X, y)

    Z = search.best_estimator_.named_steps["kpca"].transform(search.best_estimator_.named_steps["scale"].transform(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="Spectral", s=6)
    ax.set_title("Swiss roll in best kPCA space (GridSearchCV)")
    save_fig(fig, "kpca_swiss_roll_best")

    best = search.best_params_
    return {
        "best_kernel": best["kpca__kernel"],
        "best_gamma": round(float(best["kpca__gamma"]), 4),
        "best_cv_accuracy": round(float(search.best_score_), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller data/sweeps for CI")
    args = parser.parse_args()
    set_seed()
    record("pca_mnist", mnist_pca(args.quick))
    record("kernel_pca_swiss_roll", swiss_roll_kpca(args.quick))


if __name__ == "__main__":
    main()

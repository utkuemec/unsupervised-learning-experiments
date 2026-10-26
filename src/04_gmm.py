"""Gaussian Mixture Models on Olivetti faces: model selection, generation, anomaly detection.

Questions answered
  * Which covariance structure and component count minimise BIC / AIC?
  * Can the fitted density generate new (plausible) faces?
  * Does the model's log-likelihood flag manipulated images (rotated, flipped, darkened)
    as anomalies compared with genuine faces?
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, transform
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from common import SEED, face_grid, load_olivetti, record, save_fig, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    set_seed()

    X, _ = load_olivetti()
    pca = PCA(n_components=0.99, whiten=False, random_state=SEED).fit(X)
    Z = pca.transform(X)

    cov_types = ["spherical", "diag"] if args.quick else ["full", "tied", "diag", "spherical"]
    n_range = range(1, 6) if args.quick else range(1, 11)
    bic = {}
    for cov, n in itertools.product(cov_types, n_range):
        gmm = GaussianMixture(n_components=n, covariance_type=cov, random_state=SEED, reg_covar=1e-4)
        bic[(cov, n)] = gmm.fit(Z).bic(Z)
    best_cov, _ = min(bic, key=bic.get)

    bics, aics = [], []
    for n in n_range:
        gmm = GaussianMixture(n_components=n, covariance_type=best_cov, random_state=SEED, reg_covar=1e-4).fit(Z)
        bics.append(gmm.bic(Z))
        aics.append(gmm.aic(Z))
    best_n = list(n_range)[int(np.argmin(bics))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(n_range), bics, marker="o", label="BIC")
    ax.plot(list(n_range), aics, marker="s", label="AIC")
    ax.set(xlabel="components", ylabel="criterion", title=f"GMM model selection ({best_cov} covariance)")
    ax.legend()
    save_fig(fig, "gmm_bic_aic")

    gmm = GaussianMixture(n_components=best_n, covariance_type=best_cov, random_state=SEED, reg_covar=1e-4).fit(Z)
    soft = gmm.predict_proba(Z)

    # Generate new faces by sampling the density and projecting back to pixel space
    samples, _ = gmm.sample(10)
    new_faces = np.clip(pca.inverse_transform(samples), 0, 1)
    save_fig(face_grid(new_faces), "gmm_generated_faces")

    # Anomaly detection on manipulated faces
    base = X[:20].reshape(-1, 64, 64)
    edits = {
        "rotated_45": np.array([transform.rotate(f, 45) for f in base]),
        "flipped_vertical": base[:, ::-1, :],
        "darkened_gamma_3": np.array([exposure.adjust_gamma(f, 3.0) for f in base]),
    }
    normal_ll = gmm.score_samples(Z[:20])
    threshold = float(np.percentile(gmm.score_samples(Z), 2))  # flag the lowest 2% as anomalous
    anomaly = {"genuine_mean_loglik": round(float(normal_ll.mean()), 1), "threshold_p2": round(threshold, 1)}
    for name, imgs in edits.items():
        ll = gmm.score_samples(pca.transform(imgs.reshape(len(imgs), -1)))
        anomaly[name] = {
            "mean_loglik": round(float(ll.mean()), 1),
            "flagged_fraction": round(float(np.mean(ll < threshold)), 3),
        }
    examples = [base[0], edits["rotated_45"][0], edits["flipped_vertical"][0], edits["darkened_gamma_3"][0]]
    grid = face_grid(examples, titles=["genuine", "rotated", "flipped", "darkened"], n_cols=4, size=2)
    save_fig(grid, "gmm_anomaly_inputs")

    record(
        "gmm",
        {
            "pca_components_99pct": int(pca.n_components_),
            "best_covariance_type": best_cov,
            "best_n_components_bic": best_n,
            "best_n_components_aic": list(n_range)[int(np.argmin(aics))],
            "mean_max_responsibility": round(float(soft.max(axis=1).mean()), 4),
            "anomaly_detection": anomaly,
        },
    )


if __name__ == "__main__":
    main()

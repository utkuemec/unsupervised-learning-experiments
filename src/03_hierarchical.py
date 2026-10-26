"""Agglomerative (hierarchical) clustering on Olivetti faces with three distance metrics.

Questions answered
  * How do Euclidean/Ward, Manhattan/average and cosine/average linkages compare on
    silhouette score across different cluster counts?
  * With k fixed to the true number of people (40), which metric best recovers identity
    (measured with ARI / NMI against the held-back labels)?
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from common import load_olivetti, record, save_fig, set_seed

CONFIGS = {
    "euclidean_ward": {"metric": "euclidean", "linkage": "ward"},
    "manhattan_average": {"metric": "manhattan", "linkage": "average"},
    "cosine_average": {"metric": "cosine", "linkage": "average"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    set_seed()

    X, y = load_olivetti()
    ks = [20, 40, 80] if args.quick else list(range(10, 121, 10))

    sweep, identity = {}, {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, params in CONFIGS.items():
        scores = []
        for k in ks:
            labels = AgglomerativeClustering(n_clusters=k, **params).fit_predict(X)
            scores.append(silhouette_score(X, labels, metric=params["metric"]))
        sweep[name] = {"best_k": ks[int(np.argmax(scores))], "best_silhouette": round(float(max(scores)), 4)}
        ax.plot(ks, scores, marker="o", label=name)

        labels40 = AgglomerativeClustering(n_clusters=40, **params).fit_predict(X)
        identity[name] = {
            "ari": round(float(adjusted_rand_score(y, labels40)), 4),
            "nmi": round(float(normalized_mutual_info_score(y, labels40)), 4),
        }
    ax.set(xlabel="clusters", ylabel="silhouette (metric-specific)", title="Agglomerative clustering")
    ax.legend()
    save_fig(fig, "hierarchical_silhouette")

    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=40, ax=ax, no_labels=True)
    ax.set_title("Ward dendrogram (last 40 merges)")
    save_fig(fig, "hierarchical_dendrogram")

    best = max(identity, key=lambda n: identity[n]["ari"])
    record("hierarchical", {"silhouette_sweep": sweep, "k40_vs_identity": identity, "best_metric_by_ari": best})


if __name__ == "__main__":
    main()

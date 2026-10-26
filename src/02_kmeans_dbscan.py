"""K-Means and DBSCAN on Olivetti faces.

Questions answered
  * What does silhouette analysis say about the "natural" number of face clusters?
  * Can K-Means centroid distances serve as a compact feature space for face recognition,
    and how does that compare with a PCA + SVM baseline?
  * Does density-based clustering (DBSCAN) recover identities when eps is chosen from
    the k-distance curve rather than guessed?
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import SEED, face_grid, load_olivetti, record, save_fig, set_seed, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    set_seed()

    X, y = load_olivetti()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # Baseline: PCA (whitened) + RBF-SVM
    baseline = make_pipeline(
        StandardScaler(),
        PCA(n_components=100, whiten=True, random_state=SEED),
        SVC(kernel="rbf", class_weight="balanced"),
    )
    baseline_cv = cross_val_score(baseline, X_train, y_train, cv=5).mean()
    baseline.fit(X_train, y_train)
    baseline_val = baseline.score(X_val, y_val)

    # Silhouette sweep
    ks = [20, 40, 80, 120] if args.quick else list(range(10, 151, 10))
    sil = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_train)
        sil.append(silhouette_score(X_train, km.labels_))
    best_k = ks[int(np.argmax(sil))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, sil, marker="o")
    ax.axvline(best_k, ls="--", c="grey")
    ax.set(xlabel="k (clusters)", ylabel="mean silhouette", title="K-Means silhouette sweep (train split)")
    save_fig(fig, "kmeans_silhouette")

    # Cluster-distance features: each face -> distances to best_k centroids
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X_train)
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    km_cv = cross_val_score(clf, km.transform(X_train), y_train, cv=5).mean()
    clf.fit(km.transform(X_train), y_train)
    km_val = clf.score(km.transform(X_val), y_val)

    # Show 4 example clusters
    rows = []
    for c in np.argsort(np.bincount(km.labels_))[::-1][:4]:
        members = X_train[km.labels_ == c][:6]
        rows.extend(list(members) + [np.zeros(4096)] * (6 - len(members)))
    save_fig(face_grid(rows, n_cols=6), "kmeans_cluster_examples")

    # DBSCAN on a 50-D PCA space with eps from the k-distance "knee"
    Z = PCA(n_components=50, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    min_samples = 3
    dists = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1])
    eps = float(np.quantile(dists, 0.35))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Final held-out test score for the better model selected on validation
    winner = "pca_svm" if baseline_val >= km_val else "kmeans_features_svm"
    if winner == "pca_svm":
        baseline.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        test_acc = baseline.score(X_test, y_test)
    else:
        Xtv, ytv = np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
        km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xtv)
        clf.fit(km.transform(Xtv), ytv)
        test_acc = clf.score(km.transform(X_test), y_test)

    record(
        "kmeans_dbscan",
        {
            "baseline_pca_svm": {"cv_acc": round(baseline_cv, 4), "val_acc": round(baseline_val, 4)},
            "best_k_by_silhouette": best_k,
            "best_silhouette": round(float(max(sil)), 4),
            "kmeans_features_svm": {"n_features": best_k, "cv_acc": round(km_cv, 4), "val_acc": round(km_val, 4)},
            "selected_model": winner,
            "test_acc": round(float(test_acc), 4),
            "dbscan": {
                "eps": round(eps, 3),
                "min_samples": min_samples,
                "clusters": n_clusters,
                "noise_fraction": round(float(np.mean(labels == -1)), 3),
                "ari_vs_identity": round(float(adjusted_rand_score(y, labels)), 4),
            },
        },
    )


if __name__ == "__main__":
    main()

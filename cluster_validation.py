
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Load Gecko's dataset
df = pd.read_csv("youtube_tech_videos_dataset.csv")

# Make sure views and likes are valid numbers
df["views"] = pd.to_numeric(df["views"], errors="coerce")
df["likes"] = pd.to_numeric(df["likes"], errors="coerce")

df = df.dropna(subset=["views", "likes"])
df = df[(df["views"] > 0) & (df["likes"] >= 0)].copy()

# Original features
X_raw = df[["views", "likes"]]

# Log transform to reduce the influence of extreme values
X_log = np.log1p(X_raw)

# Standardize so views and likes have comparable scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)


def evaluate_kmeans(X, method):
    print(f"\n--- {method} ---")
    print("k | Silhouette | Davies-Bouldin | Inertia")

    results = []

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)

        results.append({
            "method": method,
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": db_index,
            "inertia": model.inertia_
        })

        print(
            f"{k} | {silhouette:.3f} "
            f"| {db_index:.3f} "
            f"| {model.inertia_:.2f}"
        )

    return results


# Run both experiments
raw_results = evaluate_kmeans(X_raw, "Raw views and likes")
scaled_results = evaluate_kmeans(
    X_scaled, "Log-transformed and scaled"
)

# Combine results into one table
results = pd.DataFrame(raw_results + scaled_results)

print("\nBest result by silhouette score:")
print(
    results.sort_values("silhouette", ascending=False)
    .head(1)
    .to_string(index=False)
)

# Compare cluster sizes for k = 2

for method, X in [
    ("Raw views and likes", X_raw),
    ("Log-transformed and scaled", X_scaled),
]:
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    print(f"\nCluster sizes: {method}")
    print(pd.Series(labels).value_counts().sort_index())


# Examine cluster sizes and average statistics for k = 2

for method, X in [
    ("Raw views and likes", X_raw),
    ("Log-transformed and scaled", X_scaled),
]:
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    print(f"\n--- Cluster sizes: {method} ---")
    print(pd.Series(labels).value_counts().sort_index())

    # Calculate average original views and likes per cluster
    summary = X_raw.copy()
    summary["cluster"] = labels

    print("\nAverage views and likes per cluster:")
    print(
        summary.groupby("cluster")[["views", "likes"]]
        .mean()
        .round(0)
        .to_string()
    )

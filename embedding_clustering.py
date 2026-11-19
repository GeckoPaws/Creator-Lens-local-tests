
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load Gecko's tech video dataset
df = pd.read_csv("youtube_tech_videos_dataset.csv")

# Remove missing or empty titles
df = df.dropna(subset=["title"]).copy()
df = df[df["title"].astype(str).str.strip() != ""].copy()
df = df.reset_index(drop=True)

print("Videos:", len(df))

# Load a pretrained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert video titles into numerical vectors
X = model.encode(
    df["title"].astype(str).tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embedding shape:", X.shape)

# Test different numbers of topic clusters
print("\n--- K-means Validation ---")
print("k | Cosine Silhouette | Smallest Cluster | Largest Cluster")

results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels, metric="cosine")
    sizes = pd.Series(labels).value_counts()

    results.append((k, score))

    print(
        f"{k} | {score:.3f}"
        f" | {sizes.min()}"
        f" | {sizes.max()}"
    )

# Choose the highest scoring k from this experiment
best_k = max(results, key=lambda result: result[1])[0]

print(f"\nBest k by cosine silhouette: {best_k}")

# Fit the final experimental model
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["topic_cluster"] = kmeans.fit_predict(X)

# Inspect the titles closest to each cluster center
for cluster_id in range(best_k):
    cluster_indices = [
        i for i, label in enumerate(df["topic_cluster"])
        if label == cluster_id
    ]

    center = kmeans.cluster_centers_[cluster_id]

    distances = [
        ((X[i] - center) ** 2).sum()
        for i in cluster_indices
    ]

    ranked = sorted(
        zip(cluster_indices, distances),
        key=lambda item: item[1]
    )

    print(f"\n--- Cluster {cluster_id} ---")
    print("Videos:", len(cluster_indices))
    print("Representative titles:")

    for index, _ in ranked[:6]:
        print("-", df.loc[index, "title"])


# Compare video performance across topic clusters

df["views"] = pd.to_numeric(df["views"], errors="coerce")
df["likes"] = pd.to_numeric(df["likes"], errors="coerce")

valid = df[
    (df["views"] > 0) &
    (df["likes"] >= 0)
].copy()

# Likes as a percentage of views
valid["like_rate"] = (
    valid["likes"] / valid["views"] * 100
)

summary = valid.groupby("topic_cluster").agg(
    videos=("views", "size"),
    median_views=("views", "median"),
    median_likes=("likes", "median"),
    median_like_rate=("like_rate", "median")
)

print("\n--- Topic Cluster Performance ---")
print(summary.round(2).to_string())


from pathlib import Path

# Create an output directory for modeling results
output_dir = Path("modeling_outputs")
output_dir.mkdir(exist_ok=True)

# Export video-level clustering results
columns = [
    "title",
    "topic_cluster",
    "views",
    "likes",
    "like_rate"
]

valid[columns].to_csv(
    output_dir / "video_topic_clusters.csv",
    index=False
)

# Export cluster-level performance summary
summary.round(2).to_csv(
    output_dir / "topic_cluster_summary.csv"
)

print("\nResults exported to modeling_outputs/")


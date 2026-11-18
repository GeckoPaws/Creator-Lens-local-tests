
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load the original dataset and saved topic assignments
df = pd.read_csv("youtube_tech_videos_dataset.csv")

topics = pd.read_csv(
    "modeling_outputs/video_topic_clusters.csv"
)

# Match topic assignments to the original video records
df = df.merge(
    topics[["title", "views", "likes", "topic_cluster"]],
    on=["title", "views", "likes"],
    how="left",
    validate="one_to_one"
)

# Apply the selected date range
df["published_at"] = pd.to_datetime(
    df["published_at"], utc=True
)
df["scraped_at"] = pd.to_datetime(
    df["scraped_at"], utc=True
)

start_date = pd.Timestamp("2024-11-22", tz="UTC")
end_date = pd.Timestamp("2026-11-17", tz="UTC")

df = df[
    (df["published_at"] >= start_date)
    & (df["published_at"] <= end_date)
    & (df["views"] > 0)
].copy()

# Convert video duration into seconds
df["duration_sec"] = pd.to_timedelta(
    df["duration"]
).dt.total_seconds()

# Calculate video age in days
age_days = (
    (df["scraped_at"] - df["published_at"])
    .dt.total_seconds() / 86400
).clip(lower=1)

# Create numerical performance features
df["log_views"] = np.log1p(df["views"])
df["log_likes"] = np.log1p(df["likes"])
df["log_comments"] = np.log1p(df["comments"])

df["like_rate"] = (
    df["likes"] / df["views"]
).clip(upper=1)

df["comment_rate"] = (
    df["comments"] / df["views"]
).clip(upper=1)

df["views_per_day"] = df["views"] / age_days

df["log_views_per_day"] = np.log1p(
    df["views_per_day"]
)

features = [
    "log_views",
    "log_likes",
    "log_comments",
    "like_rate",
    "comment_rate",
    "log_views_per_day"
]

# Provisional labels for the title-based clusters
topic_names = {
    0: "Mixed / Multilingual",
    1: "Web Development",
    2: "AWS / Cloud",
    3: "General Tech",
    4: "AI",
    5: "Apple / Mobile",
    6: "Python",
    7: "Gaming PCs"
}

# Create an output directory
output_dir = Path("modeling_outputs")
output_dir.mkdir(exist_ok=True)

all_results = []

# Analyze short-form and long-form videos separately
for video_type, subset in [
    ("Shorts", df[df["duration_sec"] < 180]),
    ("Long-form", df[df["duration_sec"] >= 180])
]:

    subset = subset.copy()

    if len(subset) < 10:
        print(f"\nNot enough {video_type} videos.")
        continue

    # Standardize the six performance features
    X_scaled = StandardScaler().fit_transform(
        subset[features]
    )

    # Reduce numerical features to three components
    X_pca = PCA(
        n_components=3,
        random_state=0
    ).fit_transform(X_scaled)

    # Apply four-cluster K-means
    model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=0
    )

    subset["performance_cluster"] = (
        model.fit_predict(X_pca)
    )

    # Rank performance groups by median views per day
    ranking = (
        subset.groupby("performance_cluster")["views_per_day"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    rank_map = {
        cluster: rank
        for rank, cluster in enumerate(ranking, start=1)
    }

    subset["performance_rank"] = (
        subset["performance_cluster"].map(rank_map)
    )

    # Keep videos with valid title-cluster assignments
    matched = subset.dropna(
        subset=["topic_cluster"]
    ).copy()

    matched["topic_cluster"] = (
        matched["topic_cluster"].astype(int)
    )

    matched["topic"] = (
        matched["topic_cluster"].map(topic_names)
    )

    matched["video_type"] = video_type

    # Count videos in each topic-performance combination
    counts = pd.crosstab(
        matched["topic"],
        matched["performance_rank"]
    ).reindex(
        columns=[1, 2, 3, 4],
        fill_value=0
    )

    # Calculate percentages within each topic
    percentages = (
        counts.div(counts.sum(axis=1), axis=0) * 100
    ).round(1)

    print(f"\n===== {video_type.upper()} =====")
    print("Total videos:", len(subset))
    print("Matched videos:", len(matched))

    print("\n--- Cluster Counts ---")
    print(counts.to_string())

    print("\n--- Percentage by Performance Rank ---")
    print(percentages.to_string())

    # Save separate comparison tables
    filename = video_type.lower().replace("-", "_")

    counts.to_csv(
        output_dir / f"{filename}_cluster_counts.csv"
    )

    percentages.to_csv(
        output_dir / f"{filename}_cluster_percentages.csv"
    )

    all_results.append(matched)

# Export video-level combined results
if all_results:
    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined[
        [
            "video_id",
            "title",
            "video_type",
            "topic_cluster",
            "topic",
            "performance_cluster",
            "performance_rank",
            "views",
            "likes",
            "views_per_day"
        ]
    ].to_csv(
        output_dir / "topic_vs_performance.csv",
        index=False
    )

    print("\nComparison files saved to modeling_outputs/")

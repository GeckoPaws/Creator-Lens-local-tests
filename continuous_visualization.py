
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# Load combined topic and performance assignments
output_dir = Path("modeling_outputs")

df = pd.read_csv(
    output_dir / "topic_vs_performance.csv"
)

df = df.reset_index(drop=True)

# Calculate continuous performance metrics
df["like_rate"] = (
    df["likes"] / df["views"] * 100
)

df["log_views_per_day"] = np.log1p(
    df["views_per_day"]
)

topic_order = [
    "Gaming PCs",
    "Apple / Mobile",
    "General Tech",
    "Web Development",
    "Mixed / Multilingual",
    "Python",
    "AI",
    "AWS / Cloud"
]

formats = ["Shorts", "Long-form"]

colors = {
    topic: plt.get_cmap("tab10")(i)
    for i, topic in enumerate(topic_order)
}


# ==========================================
# 1. Continuous performance distributions
# ==========================================

def create_distribution(metric, xlabel, title, filename):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(17, 7),
        sharey=False,
        sharex=True
    )

    for ax, video_format in zip(axes, formats):

        subset = df[
            df["video_type"] == video_format
        ]

        values = [
            subset.loc[
                subset["topic"] == topic, metric
            ].dropna().values
            for topic in topic_order
        ]

        boxes = ax.boxplot(
            values,
            orientation="horizontal",
            patch_artist=True,
            showfliers=True,
            medianprops={
                "color": "black",
                "linewidth": 1.5
            },
            flierprops={
                "markersize": 3,
                "alpha": 0.4
            }
        )

        # Apply consistent topic colors
        for box, topic in zip(
            boxes["boxes"], topic_order
        ):
            box.set_facecolor(colors[topic])
            box.set_alpha(0.6)

        # Display independent sample sizes for each format
        labels = [
            f"{topic} (n={len(data)})"
            for topic, data in zip(topic_order, values)
        ]

        ax.set_yticks(
            range(1, len(topic_order) + 1)
        )

        ax.set_yticklabels(labels)

        ax.set_title(
            f"{video_format} ({len(subset)} videos)",
            fontsize=13
        )

        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        # Place the first topic at the top
        ax.invert_yaxis()

        # Display original units on the transformed axis
        if metric == "log_views_per_day":
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: f"{np.expm1(x):,.0f}"
                )
            )

    fig.suptitle(title, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(
        output_dir / filename,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


create_distribution(
    "log_views_per_day",
    "Average Views per Day (log-transformed scale)",
    "Distribution of Video Reach Across Topics",
    "views_per_day_distribution.png"
)

create_distribution(
    "like_rate",
    "Likes / Views (%)",
    "Distribution of Like Rates Across Topics",
    "like_rate_distribution.png"
)


# ==========================================
# 2. Continuous performance summary
# ==========================================

summary = df.groupby(
    ["video_type", "topic"]
).agg(

    video_count=("views_per_day", "size"),

    median_views_per_day=(
        "views_per_day", "median"
    ),

    q1_views_per_day=(
        "views_per_day",
        lambda x: x.quantile(0.25)
    ),

    q3_views_per_day=(
        "views_per_day",
        lambda x: x.quantile(0.75)
    ),

    median_like_rate=(
        "like_rate", "median"
    ),

    q1_like_rate=(
        "like_rate",
        lambda x: x.quantile(0.25)
    ),

    q3_like_rate=(
        "like_rate",
        lambda x: x.quantile(0.75)
    )

).reset_index()

summary.round(4).to_csv(
    output_dir / "continuous_performance_summary.csv",
    index=False
)


# ==========================================
# 3. Generate semantic coordinates
# ==========================================

print("\nGenerating sentence embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["title"].astype(str).tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)

# Fit a shared PCA projection across both video formats
pca = PCA(
    n_components=2,
    random_state=42
)

coordinates = pca.fit_transform(embeddings)

df["semantic_x"] = coordinates[:, 0]
df["semantic_y"] = coordinates[:, 1]

pc1_variance = pca.explained_variance_ratio_[0]
pc2_variance = pca.explained_variance_ratio_[1]

explained = (
    pc1_variance + pc2_variance
)

print(
    f"Semantic PC1 variance: {pc1_variance:.2%}"
)

print(
    f"Semantic PC2 variance: {pc2_variance:.2%}"
)

print(
    f"Total 2D PCA variance retained: {explained:.2%}"
)


# ==========================================
# 4. Calculate continuous marker sizes
# ==========================================

reach = df["log_views_per_day"]

reach_min = reach.min()
reach_max = reach.max()
reach_range = reach_max - reach_min


def calculate_marker_size(views_per_day):

    log_reach = np.log1p(views_per_day)

    if reach_range > 0:

        normalized = (
            (log_reach - reach_min) / reach_range
        )

        normalized = np.clip(
            normalized, 0, 1
        )

        return 15 + 185 * normalized

    return 50


df["marker_size"] = (
    df["views_per_day"].apply(
        calculate_marker_size
    )
)


# ==========================================
# 5. Create semantic performance maps
# ==========================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(17, 9),
    sharex=True,
    sharey=True
)

for ax, video_format in zip(axes, formats):

    subset = df[
        df["video_type"] == video_format
    ]

    for topic in topic_order:

        group = subset[
            subset["topic"] == topic
        ]

        # Plot larger points first to reduce obstruction
        group = group.sort_values(
            "marker_size",
            ascending=False
        )

        ax.scatter(
            group["semantic_x"],
            group["semantic_y"],
            s=group["marker_size"],
            color=colors[topic],
            alpha=0.55,
            edgecolors="none",
            label=topic
        )

    ax.set_title(
        f"{video_format} ({len(subset)} videos)",
        fontsize=13
    )

    ax.set_xlabel(
        f"Semantic PC1 ({pc1_variance:.1%} variance)"
    )

    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)

    ax.set_aspect("equal", adjustable="box")

axes[0].set_ylabel(
    f"Semantic PC2 ({pc2_variance:.1%} variance)"
)


# ==========================================
# 6. Create topic and marker-size legends
# ==========================================

topic_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        color=colors[topic],
        label=topic,
        markersize=8
    )
    for topic in topic_order
]

fig.legend(
    handles=topic_handles,
    title="Semantic Topic Cluster",
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.065)
)

# Use observed reach percentiles as size references
reference_values = np.unique(
    np.round(
        df["views_per_day"].quantile(
            [0.25, 0.50, 0.90]
        ).values
    )
)

size_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        color="gray",
        alpha=0.65,
        markersize=np.sqrt(
            calculate_marker_size(value)
        ),
        label=f"{value:,.0f} views/day"
    )
    for value in reference_values
]

fig.legend(
    handles=size_handles,
    title="Marker Size Reference",
    loc="lower center",
    ncol=len(size_handles),
    bbox_to_anchor=(0.5, 0.005)
)

fig.suptitle(
    "Semantic Content and Performance Map\n"
    "Color = Topic | Point Area = Log-scaled Reach\n"
    f"Total PCA Variance Retained: {explained:.1%}",
    fontsize=15
)

fig.tight_layout(
    rect=[0, 0.19, 1, 0.89]
)

fig.savefig(
    output_dir / "semantic_performance_map.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)


# ==========================================
# 7. Export dashboard-ready video data
# ==========================================

export_columns = [
    "video_id",
    "title",
    "video_type",
    "topic_cluster",
    "topic",
    "performance_cluster",
    "performance_rank",
    "views",
    "likes",
    "views_per_day",
    "like_rate",
    "log_views_per_day",
    "semantic_x",
    "semantic_y",
    "marker_size"
]

df[export_columns].to_csv(
    output_dir / "dashboard_video_performance.csv",
    index=False
)


# ==========================================
# 8. Print final results
# ==========================================

print("\n--- Continuous Performance Summary ---")

print(
    summary.round(2).to_string(index=False)
)

print("\n--- Export Summary ---")

print("Total videos:", len(df))

print(
    "Shorts:",
    sum(df["video_type"] == "Shorts")
)

print(
    "Long-form:",
    sum(df["video_type"] == "Long-form")
)

print("\nFiles generated:")

print("- views_per_day_distribution.png")
print("- like_rate_distribution.png")
print("- semantic_performance_map.png")
print("- continuous_performance_summary.csv")
print("- dashboard_video_performance.csv")

print("\nContinuous visualizations and data saved.")

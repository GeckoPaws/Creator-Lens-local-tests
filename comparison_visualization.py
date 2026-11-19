
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = Path("modeling_outputs")

# Load topic and performance cluster counts
shorts = pd.read_csv(
    output_dir / "shorts_cluster_counts.csv",
    index_col=0
)

longform = pd.read_csv(
    output_dir / "long_form_cluster_counts.csv",
    index_col=0
)

shorts.columns = shorts.columns.astype(int)
longform.columns = longform.columns.astype(int)

ranks = [1, 2, 3, 4]

shorts = shorts.reindex(columns=ranks, fill_value=0)
longform = longform.reindex(columns=ranks, fill_value=0)

# Calculate percentages within each topic
shorts_pct = shorts.div(shorts.sum(axis=1), axis=0) * 100
longform_pct = longform.div(longform.sum(axis=1), axis=0) * 100

# Use a consistent topic order across both formats
topic_order = shorts_pct[1].sort_values(
    ascending=False
).index

shorts = shorts.reindex(topic_order)
longform = longform.reindex(topic_order)

shorts_pct = shorts_pct.reindex(topic_order)
longform_pct = longform_pct.reindex(topic_order)

# ==========================================
# 1. Topic vs. performance heatmap
# ==========================================

def create_heatmap(counts, percentages, name, filename):

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        percentages.values,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_xticks(range(4))
    ax.set_xticklabels([
        "Rank 1\nHighest Reach",
        "Rank 2",
        "Rank 3",
        "Rank 4\nLowest Reach"
    ])

    ax.set_yticks(range(len(percentages)))
    ax.set_yticklabels(percentages.index)

    # Display proportions and video counts
    for i in range(len(percentages)):
        for j in range(4):

            pct = percentages.iloc[i, j]
            count = counts.iloc[i, j]

            ax.text(
                j, i,
                f"{pct:.1f}%\n(n={count})",
                ha="center",
                va="center",
                color="white" if pct > 50 else "black",
                fontsize=10
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Percentage within topic (%)"
    )

    ax.set_title(
        f"Topic vs. Performance: {name}\n"
        f"{int(counts.values.sum())} videos"
    )

    ax.set_xlabel("Performance Rank")
    ax.set_ylabel("Topic Cluster")

    fig.tight_layout()

    fig.savefig(
        output_dir / filename,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


create_heatmap(
    shorts,
    shorts_pct,
    "Shorts",
    "shorts_topic_performance_heatmap.png"
)

create_heatmap(
    longform,
    longform_pct,
    "Long-form",
    "longform_topic_performance_heatmap.png"
)

# ==========================================
# 2. Highest-reach comparison
# ==========================================

x = np.arange(len(topic_order))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(
    x - width / 2,
    shorts_pct[1],
    width,
    label="Shorts",
    color="steelblue"
)

bars2 = ax.bar(
    x + width / 2,
    longform_pct[1],
    width,
    label="Long-form",
    color="darkorange"
)

ax.bar_label(bars1, fmt="%.1f%%", fontsize=8)
ax.bar_label(bars2, fmt="%.1f%%", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(
    topic_order,
    rotation=30,
    ha="right"
)

ax.set_ylim(0, 100)
ax.set_ylabel("Videos in highest-reach cluster (%)")
ax.set_title("Highest-Reach Videos by Topic and Format")

ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    output_dir / "highest_reach_by_topic.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================
# 3. Dashboard-ready aggregate dataset
# ==========================================

records = []

for video_format, counts, percentages in [
    ("Shorts", shorts, shorts_pct),
    ("Long-form", longform, longform_pct)
]:

    for topic in counts.index:

        total = int(counts.loc[topic].sum())

        for rank in ranks:

            records.append({
                "video_format": video_format,
                "topic": topic,
                "performance_rank": rank,
                "video_count": int(
                    counts.loc[topic, rank]
                ),
                "topic_total_videos": total,
                "percentage_within_topic": round(
                    percentages.loc[topic, rank], 2
                )
            })

dashboard_df = pd.DataFrame(records)

dashboard_df.to_csv(
    output_dir / "dashboard_topic_performance.csv",
    index=False
)

# Export a simplified highest-reach comparison
highest_reach = pd.DataFrame({
    "topic": topic_order,
    "shorts_total": shorts.sum(axis=1).values,
    "shorts_rank1_count": shorts[1].values,
    "shorts_rank1_pct": shorts_pct[1].round(2).values,
    "longform_total": longform.sum(axis=1).values,
    "longform_rank1_count": longform[1].values,
    "longform_rank1_pct": longform_pct[1].round(2).values
})

highest_reach.to_csv(
    output_dir / "dashboard_highest_reach.csv",
    index=False
)

print("\n--- Visualization Summary ---")
print("Shorts:", int(shorts.values.sum()))
print("Long-form:", int(longform.values.sum()))
print("Dashboard records:", len(dashboard_df))

print("\n--- Highest-Reach Comparison ---")
print(highest_reach.to_string(index=False))

print("\nVisualizations and dashboard data saved.")

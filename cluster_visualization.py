
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load the results from our embedding clustering experiment
output_dir = Path("modeling_outputs")

df = pd.read_csv(
    output_dir / "topic_cluster_summary.csv"
)

# Descriptive labels based on representative video titles
# These are provisional, not verified topic classifications.
topic_labels = {
    0: "Mixed / Multilingual",
    1: "Web Development",
    2: "AWS / Cloud",
    3: "General Tech / Careers",
    4: "AI / AI Agents",
    5: "Apple / Mobile",
    6: "Python",
    7: "Gaming PCs"
}

df["topic"] = df["topic_cluster"].map(topic_labels)

# --------------------------
# Chart 1: Median views
# --------------------------

views_df = df.sort_values("median_views")

plt.figure(figsize=(10, 6))

plt.barh(
    views_df["topic"],
    views_df["median_views"],
    color="steelblue"
)

plt.xlabel("Median Views")
plt.title("Median YouTube Views by Topic Cluster")
plt.tight_layout()

plt.savefig(
    output_dir / "median_views_by_topic.png",
    dpi=200
)

plt.show()
plt.close()

# --------------------------
# Chart 2: Median like rate
# --------------------------

likes_df = df.sort_values("median_like_rate")

plt.figure(figsize=(10, 6))

plt.barh(
    likes_df["topic"],
    likes_df["median_like_rate"],
    color="mediumseagreen"
)

plt.xlabel("Median Likes / Views (%)")
plt.title("Median Like Rate by Topic Cluster")
plt.tight_layout()

plt.savefig(
    output_dir / "median_like_rate_by_topic.png",
    dpi=200
)

plt.show()
plt.close()

print("\nVisualizations saved to modeling_outputs/")

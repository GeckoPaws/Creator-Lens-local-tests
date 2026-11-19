import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("youtube_tech_videos_dataset.csv")

# Remove rows without titles
df = df.dropna(subset=["title"]).copy()

# Convert video titles into numerical vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["title"].astype(str))

# Group videos into 5 topic clusters (initial experiment)
model = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=10
)

df["topic_cluster"] = model.fit_predict(X)

# Display representative terms for each cluster
terms = vectorizer.get_feature_names_out()

for cluster_id in range(8):
    center = model.cluster_centers_[cluster_id]
    top_indices = center.argsort()[-8:][::-1]

    print(f"\nCluster {cluster_id}")
    print("Top terms:", ", ".join(terms[top_indices]))

    examples = df[df["topic_cluster"] == cluster_id]["title"].head(5)

    for title in examples:
        print("-", title)


from sklearn.metrics import silhouette_score

print("\n--- Cluster Sizes ---")
print(df["topic_cluster"].value_counts().sort_index())

score = silhouette_score(
    X,
    df["topic_cluster"],
    metric="cosine"
)

print(f"\nCosine Silhouette Score: {score:.3f}")


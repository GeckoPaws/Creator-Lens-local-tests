# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

'''
Finds trending video topics in a time range.
1. Build engagement features (not just views and likes)
2. Log-transform + scale them, then reduce with PCA
3. KMeans on the PCA components (k chosen with elbow + silhouette)
4. Plot clusters in 3D
5. TF-IDF on titles to find the dominant keywords in each cluster
'''

# %%
df = pd.read_csv("youtube_tech_videos_dataset.csv")
df["published_at"] = pd.to_datetime(df["published_at"])

# Prompt for a timeframe. Re-asks until valid. Blank = use the whole dataset
def ask_date(prompt, default):
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            return datetime.strptime(raw, "%Y/%m/%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("Invalid date, use yyyy/mm/dd")

date_start = ask_date("Enter start date (yyyy/mm/dd, blank = earliest): ", df["published_at"].min().to_pydatetime())
date_end = ask_date("Enter end date (yyyy/mm/dd, blank = latest): ", df["published_at"].max().to_pydatetime())

if date_start > date_end:
    sys.exit("start date must be before end date")

# .copy() avoids pandas' SettingWithCopyWarning on the columns added below
# Videos with 0 views have undefined engagement rates, so they're dropped
df_time_range = df[(df["published_at"] >= date_start) & (df["published_at"] <= date_end) & (df["views"] > 0)].copy()
print(f"{len(df_time_range)} videos in range")

# %%
df_time_range["duration_sec"] = pd.to_timedelta(df_time_range["duration"]).dt.total_seconds()

# Engagement features. Rates and velocity matter more than raw counts
scraped = pd.to_datetime(df_time_range["scraped_at"]).dt.tz_localize("UTC")
age_days = ((scraped - df_time_range["published_at"]).dt.total_seconds() / 86400).clip(lower=1)

df_time_range["log_views"] = np.log1p(df_time_range["views"])
df_time_range["log_likes"] = np.log1p(df_time_range["likes"])
df_time_range["log_comments"] = np.log1p(df_time_range["comments"])

# Rates are capped at 1 so glitchy rows (likes > views) can't become their own cluster
df_time_range["like_rate"] = (df_time_range["likes"] / df_time_range["views"]).clip(upper=1)
df_time_range["comment_rate"] = (df_time_range["comments"] / df_time_range["views"]).clip(upper=1)
df_time_range["log_views_per_day"] = np.log1p(df_time_range["views"] / age_days)

FEATURES = ["log_views", "log_likes", "log_comments", "like_rate", "comment_rate", "log_views_per_day"]

# Shorts and longform have different engagement behavior, so they're clustered separately
df_shorts = df_time_range[df_time_range["duration_sec"] < 180].copy()
df_longform = df_time_range[df_time_range["duration_sec"] >= 180].copy()
print(f"shorts: {len(df_shorts)}, longform: {len(df_longform)}")

# %%
def cluster_videos(data, name, k_max=8, n_components=3):
    """Scale -> PCA -> pick k -> KMeans. Returns data with 'cluster' and PC columns."""
    data = data.copy()

    # Log transform handles the heavy skew; scaling puts all features on equal footing
    X = StandardScaler().fit_transform(data[FEATURES])

    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)
    print(f"\n[{name}] variance explained by PCs: {np.round(pca.explained_variance_ratio_, 3)} "
          f"(total {pca.explained_variance_ratio_.sum():.2f})")

    # Can't have more clusters than samples - 1
    K_values = list(range(2, min(k_max, len(data) - 1) + 1))
    inertias, sils = [], []
    for k in K_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_pca)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_pca, km.labels_))

    # Elbow, with the silhouette score as a sanity check
    kl = KneeLocator(K_values, inertias, curve="convex", direction="decreasing")
    k_elbow = kl.elbow
    k_sil = K_values[int(np.argmax(sils))]
    k = k_elbow if k_elbow is not None else k_sil
    print(f"[{name}] elbow k = {k_elbow}, best silhouette k = {k_sil}, using k = {k}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(K_values, inertias, "bo-")
    if k_elbow:
        ax[0].axvline(k_elbow, ls="--", c="tab:red", label=f"elbow = {k_elbow}")
        ax[0].legend()
    ax[0].set(xlabel="k", ylabel="Inertia", title=f"{name}: elbow")
    ax[1].plot(K_values, sils, "go-")
    ax[1].set(xlabel="k", ylabel="Silhouette", title=f"{name}: silhouette")
    plt.tight_layout()
    plt.savefig(f"elbow_{name}.png", dpi=120)
    plt.show()

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_pca)
    data["cluster"] = km.labels_
    for i in range(n_components):
        data[f"PC{i+1}"] = X_pca[:, i]

    return data, km, pca

def plot_3d(data, km, name):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(data["PC1"], data["PC2"], data["PC3"], c=data["cluster"], cmap="tab10", s=25, alpha=0.8)
    c = km.cluster_centers_
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], marker="X", c="red", s=200, edgecolor="k", label="centroids")
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=f"{name}: KMeans clusters in PCA space")
    ax.legend(*sc.legend_elements(), title="cluster", loc="upper left")
    plt.savefig(f"clusters3d_{name}.png", dpi=120)
    plt.show()

# %%
def top_keywords(data, name, top_n=10):
    """Rank clusters by trendiness, then show the most characteristic title terms of each."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+]+\b")
    tfidf = vec.fit_transform(data["title"].fillna(""))
    terms = np.array(vec.get_feature_names_out())
    global_mean = np.asarray(tfidf.mean(axis=0)).ravel()

    # Cluster summary, sorted so the most "trending" cluster is first
    summary = (data.groupby("cluster")
               .agg(n=("title", "size"),
                    median_views=("views", "median"),
                    median_views_per_day=("log_views_per_day", lambda s: np.expm1(s).median()),
                    like_rate=("like_rate", "mean"),
                    comment_rate=("comment_rate", "mean"))
               .sort_values("median_views_per_day", ascending=False))
    print(f"\n===== {name}: cluster summary (best first) =====")
    print(summary.round(4))

    for cl in summary.index:
        idx = np.where(data["cluster"].values == cl)[0]
        cl_mean = np.asarray(tfidf[idx].mean(axis=0)).ravel()

        # Subtracting the global mean favors words that are distinctive to this cluster
        score = cl_mean - global_mean
        best = terms[np.argsort(score)[::-1][:top_n]]
        print(f"\nCluster {cl} ({len(idx)} videos): {', '.join(best)}")

        top_vids = data.iloc[idx].nlargest(3, "views")["title"].tolist()
        for t in top_vids:
            print(f"   - {t}")

# %%
# Run the whole pipeline on shorts and longform
for name, subset in [("shorts", df_shorts), ("longform", df_longform)]:
    if len(subset) < 10:
        print(f"\n[{name}] only {len(subset)} videos in range, too few to cluster. Widen the date range.")
        continue
    clustered, km, pca = cluster_videos(subset, name)
    plot_3d(clustered, km, name)
    top_keywords(clustered, name)
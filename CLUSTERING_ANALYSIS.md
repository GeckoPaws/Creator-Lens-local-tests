
# CreatorLens: K-means Clustering Analysis

## Objective

Explore whether K-means clustering can identify meaningful groups of YouTube videos based on popularity and title content, and compare video performance across those groups.

Dataset: 1,300 videos from 130 selected technology channels.

## 1. Views and Likes Clustering

Initial experiments applied K-means to raw views and likes.

At k=2, the raw model produced:
- Silhouette score: 0.997
- Cluster sizes: 1,293 and 1

The model primarily isolated a single video with approximately 79 million views rather than identifying useful popularity groups.

After applying log1p transformation and StandardScaler:
- Silhouette score: 0.551
- Cluster sizes: 728 and 566

This suggests that transforming highly skewed numerical features can reduce the influence of extreme outliers.

These results used 1,294 videos with valid numerical features.

## 2. TF-IDF Title Clustering

Used TF-IDF vectorization with K-means to group videos according to shared words and phrases in their titles.

At k=8:
- Cosine silhouette score: 0.043
- Largest cluster: 845 videos
- Smallest cluster: 43 videos

Recognizable groups included Python, gaming PCs, AI, JavaScript, and tech news.

However, approximately 65% of videos were assigned to one mixed-topic cluster, limiting the usefulness of this approach.

## 3. Sentence Embedding Clustering

Used the pretrained all-MiniLM-L6-v2 model to convert video titles into 384-dimensional sentence embeddings.

Applied K-means and evaluated k=2 through k=8 using cosine silhouette scores and cluster sizes.

At k=8:
- Cosine silhouette score: 0.080
- Largest cluster: 301 videos
- Smallest cluster: 90 videos

The resulting clusters included:
- Web development / JavaScript
- AWS / cloud computing
- Artificial intelligence
- Apple / mobile development
- Python
- Gaming PC builds
- Two mixed-topic groups

Sentence embeddings produced a higher cosine silhouette score and a less concentrated cluster distribution than TF-IDF at the same k.

However, the score remains low, so the clusters are not strongly separated. k=8 was the best among the tested values, not necessarily the optimal number of clusters.

## 4. Engagement Analysis

Using the sentence embedding clusters, calculated median views, median likes, and median like-to-view rates.

| Cluster | Approximate Topic | Median Views | Median Like Rate |
|---------|-------------------|-------------:|-----------------:|
| 0 | Mixed / Multilingual | 1,917 | 3.25% |
| 1 | Web Development | 3,704 | 3.09% |
| 2 | AWS / Cloud | 1,154.5 | 2.89% |
| 3 | General Tech / Careers | 4,156 | 4.01% |
| 4 | AI / AI Agents | 988 | 3.21% |
| 5 | Apple / Mobile | 8,345 | 3.40% |
| 6 | Python | 1,491 | 3.33% |
| 7 | Gaming PCs | 34,542 | 3.58% |

The gaming PC group had the highest median views, while the mixed general tech/careers group had the highest median like-to-view rate.

The rate measures likes divided by views, not total engagement including comments. Six videos were excluded from this analysis because of invalid numerical values.

## 5. Limitations

- The dataset contains selected technology creators, not a representative sample of YouTube.
- Channel audience sizes, video ages, and creator differences may influence the performance comparisons.
- Some topic clusters are mixed or difficult to interpret.
- The embedding model is primarily designed for English, while some video titles are in other languages.
- Repeated titles or content from the same channels may influence clustering.
- Cluster numbers are arbitrary, and topic labels were manually inferred.
- The findings show associations, not causal effects.

## 6. Conclusion

Sentence embeddings are the more promising title representation among the configurations tested.

They produced more recognizable topic groupings across the dataset and improved cosine silhouette scores compared with TF-IDF.

However, clustering quality remains limited, and the method should be treated as an exploratory approach rather than a finalized topic classification system.

## 7. Possible Next Steps

- Evaluate multilingual sentence embeddings.
- Investigate whether clusters are dominated by individual channels.
- Compare performance relative to each channel's typical views.
- Explore additional k values and topic coherence.
- Integrate validated clustering results into the CreatorLens dashboard.

## Scripts

- `cluster_validation.py`: Numerical K-means preprocessing and validation
- `title_clustering.py`: TF-IDF title clustering
- `embedding_clustering.py`: Sentence embedding clustering and engagement analysis
- `cluster_visualization.py`: Performance visualizations

## Running the Experiments

Install the necessary dependencies, including pandas, numpy, scikit-learn, matplotlib, and sentence-transformers.

Run:

```bash
python cluster_validation.py
python title_clustering.py
python embedding_clustering.py
python cluster_visualization.py
```

The scripts use `youtube_tech_videos_dataset.csv` from the repository.

Outputs from the embedding and visualization scripts are saved in `modeling_outputs/`.
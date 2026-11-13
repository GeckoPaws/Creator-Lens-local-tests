# %%
import pandas as pd
from datetime import datetime, timezone
import sys
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
'''
This is a test to try and find trends among videos in the tech videos dataset
I will be finding whats most trending between a specific time range
The user will select a time range and then pick what kind of thrend they want to see
options will be most liked or most viewed

Style choices:
One line before and after code blocks
One line before comments
'''

# %%
df = pd.read_csv(r'youtube_tech_videos_dataset.csv')
df["published_at"] = pd.to_datetime(df["published_at"])

#prompting the user for a desired timeframe to search trends and validating the date
date_start = input("Enter start date: (yyyy/mm/dd)")
date_end = input("Enter end date: (yyyy/mm/dd)")

if(date_start == '1' and date_end == '1'):
    date_start = '2024/01/01'
    date_end = '2024/12/31'

try:
    date_start = datetime.strptime(date_start, "%Y/%m/%d").replace(tzinfo = timezone.utc)
    date_end =datetime.strptime(date_end, "%Y/%m/%d").replace(tzinfo = timezone.utc)

    print("start date: ", date_start)
    print("end date: ", date_end)
    print(type(date_end), type(date_start))

except ValueError:
    print("time conversion unsuccessful")

if(date_start > date_end):
    print("start date must be before end date")
    sys.exit()

df_time_range = df[(df['published_at'] >= date_start) & (df['published_at'] <= date_end)]

# %%
# Transformed duration into readable seconds format
df_time_range['duration'] = pd.to_timedelta(df['duration']).dt.total_seconds()

#Dataframe has to be split into two more subgroups, shorts and longform
#because shorts have different egnagement rates than longform vids
df_shorts = df_time_range[df_time_range["duration"] < 180]
df_longform = df_time_range[df_time_range["duration"] >= 180]

# %%
#this is just to test that the df being made correctly
print("2024")
print(df_time_range["published_at"].head())

print("original")
print(df["published_at"].head)

# %%
### TASKS ###
# next i want to cluster the data by likes and views
# and see what videos are in the most liked group. Then I'll 
# try to assign them a label based on the majority of the titles
# probably need to compare how similar the titles are to each
# other using some kind of word vector thing

# %%
#### YOUTUBE SHORTS ####
likesShorts = df_shorts[['views', 'likes']]
inertias = []
K_values = range(1, 11)
for k in K_values:
    km = KMeans(n_clusters=k)
    km.fit(likesShorts)
    inertias.append(km.inertia_)
    
# Determining the elbow and visualization
kl_short = KneeLocator(list(K_values), inertias, curve="convex", direction="decreasing")
print(f"Optimal k: {kl_short.elbow}")
    
plt.figure(figsize=(7, 4))
plt.plot(K_values, inertias, "bo-")
plt.vlines(kl_short.elbow, plt.ylim()[0], plt.ylim()[1], linestyles="--", label=f"elbow = {kl_short.elbow}", colors='tab:red')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method Visualization")
plt.legend()
plt.show()

# %%
# Clustering with chosen K
kmeans_shorts = KMeans(n_clusters=kl_short.elbow, init='k-means++', random_state=0).fit(likesShorts)
print("\nCluster Centers: ")
print(f"Views: {kmeans_shorts.cluster_centers_[:,0]}")
print(f"Likes: {kmeans_shorts.cluster_centers_[:,1]}")

sns.scatterplot(data=likesShorts, x="views", y="likes", hue=kmeans_shorts.labels_)
plt.scatter(kmeans_shorts.cluster_centers_[:,0], kmeans_shorts.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()

#%%
# Applying cluster labels to dataframe
df_shorts['cluster'] = kmeans_shorts.fit_predict(likesShorts)
df_shorts.head()

#%% 
#### LONGFORM VIDEOS ####
likesLong = df_longform[['views', 'likes']]
inertias = []
K_values = range(1, 11)
for k in K_values:
    km = KMeans(n_clusters=k)
    km.fit(likesLong)
    inertias.append(km.inertia_)
    
# Determining the elbow and visualization
kl_long = KneeLocator(list(K_values), inertias, curve="convex", direction="decreasing")
print(f"Optimal k: {kl_long.elbow}")
    
plt.figure(figsize=(7, 4))
plt.plot(K_values, inertias, "bo-")
plt.vlines(kl_long.elbow, plt.ylim()[0], plt.ylim()[1], linestyles="--", label=f"elbow = {kl_long.elbow}", colors='tab:red')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method Visualization")
plt.legend()
plt.show()

# %%
# Clustering with chosen K
kmeans_long = KMeans(n_clusters=kl_long.elbow, init='k-means++', random_state=0).fit(likesLong)
print("\nCluster Centers: ")
print(f"Views: {kmeans_long.cluster_centers_[:,0]}")
print(f"Likes: {kmeans_long.cluster_centers_[:,1]}")

sns.scatterplot(data=likesLong, x="views", y="likes", hue=kmeans_long.labels_)
plt.scatter(kmeans_long.cluster_centers_[:,0], kmeans_long.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()

#%%
# Applying cluster labels to dataframe
df_longform['cluster'] = kmeans_long.fit_predict(likesLong)
df_longform.head()
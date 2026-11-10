# %%
import pandas as pd
from datetime import datetime, timezone
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
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
df = pd.read_csv(r'C:\Users\isabe\Documents\MyDocuments\Coding\CreatorLen_local\youtube_tech_videos_dataset.csv')
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

#Dataframe has to be split into two more subgroups, shorts and longform
#because shorts have different egnagement rates than longform vids
df_shorts = df_time_range[df_time_range["duration_readable"] < 3 ]
df_longform = df_time_range[df_time_range["duration_readable"] >= 3]

# %%
#this is just to test that the df being made correctly
print("2024")
print(df_time_range["published_at"].head())

print("original")
print(df["published_at"].head)
# %%
# next i want to cluster the data by likes and views
# and see what videos are in the most liked group. Then I'll 
# try to assign them a label based on the majority of the titles
# probably need to compare how similar the titles are to each
# other using some kind of word vector thing
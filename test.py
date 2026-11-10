import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

df = pd.read_csv('youtube_tech_videos_dataset.csv')

#prompting the user for a desired timeframe to search trends and validating the date
time_start = input("Enter start date: (yyyy/mm/dd)")
time_end = input("Enter end date: (yyyy/mm/dd)")

try:
    time_start = datetime.strptime(time_start, "%Y/%m/%d")
    time_end =datetime.strptime(time_end, "%Y/%m/%d")

    print("start time: ", time_start)
    print("end time: ", time_end)
    print(type(time_end), type(time_start))
except ValueError:
    print("time conversion unsuccessful")


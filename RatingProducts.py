"""
#################################
RATING PRODUCTS
#################################
"""

# - Average
# - Time-Based-Weighted Average
# - User-Based-Weighted Average
# - Weighted Rating

# (50+ Hours) Python A to Z: Data Science and Machine Learning
# Points: 4.8(4.764925)
# Total Ratings: 4611
# Point Percentages: 75, 20, 4, 1, <1
# Approximate Numerical Equivalents: 3548, 922, 184, 46, 6


###############
#Application: Calculating User and Time Oriented Course Notes
###############

import pandas as pd
import math
import scipy.stats as st
from pandas.core.interchange import dataframe
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#region Average
df = pd.read_csv('datasets/course_reviews.csv')
df.head()
df["Rating"].value_counts()
df["Questions Asked"].value_counts()
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                          "Rating": "mean"})
df.head()

# Average Rating
df["Rating"].mean()

#endregion

#region Time-Based Weighted Average

df.info()
df.head()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Timestamp"].max()

current_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (current_date - df["Timestamp"]).dt.days
df.head()

df[df["days"] <= 30].head()
df[df["days"] <= 30].count()

df.loc[df["days"] <= 30, "Rating"].mean()
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
df.loc[df["days"] > 180, "Rating"].mean()

df.loc[df["days"] <= 30, "Rating"].mean() * 0.28 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 0.26 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 0.24 + \
df.loc[df["days"] > 180, "Rating"].mean() * 0.22


def time_weighted_average(dataframe, w1=0.28, w2=0.26, w3=0.24, w4=0.22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 + \
        dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 + \
        dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 + \
        dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4


time_weighted_average(df)

time_weighted_average(df, 0.30, 0.26, 0.22, 0.22)
#endregion

#region User-Based Weighted Average
df.head()
df.groupby("Progress").agg({"Rating": "mean"})

df.loc[df["Progress"] <=10, "Rating"].mean() * 22/100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24/100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26/100 + \
df.loc[df["Progress"] > 75, "Rating"].mean() * 28/100

def user_weighted_average(dataframe, w1=20, w2=24, w3=26, w4=30):
    return dataframe.loc[dataframe["Progress"] <=10, "Rating"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
        dataframe.loc[dataframe["Progress"] > 75, "Rating"].mean() * w4 / 100

user_weighted_average(df)
#endregion

#region Weighted Rating

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_weighted_average(dataframe) * time_w/100 + \
        user_weighted_average(dataframe) * user_w/100

course_weighted_rating(df)
course_weighted_rating(df, 40, 60)
#endregion
"""
#################################
SORTING PRODUCTS
#################################
"""

###############
#Application: Sorting Courses
###############

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('datasets/product_sorting.csv')
print(df.shape)
df.head()

#region Sorting by Rating
df.sort_values(by="rating", ascending=False).head(20)
#endregion

#region Sorting by Comment or Purchase Count
df.sort_values(by="purchase_count", ascending=False).head(20)
df.sort_values(by="commment_count", ascending=False).head(20)
#endregion

#region Sorting by Rating, Comment and Purchase
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df["commment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

(df["commment_count_scaled"] * 0.32 +
 df["purchase_count_scaled"] * 0.26 +
 df["rating"] * 0.42)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return  df["commment_count_scaled"] * w1/100 + \
        df["purchase_count_scaled"] * w2/100 + \
        df["rating"] * w3/100

df["weighted_sorting_score"] = weighted_sorting_score(df)
df.sort_values(by="weighted_sorting_score", ascending=False).head(20)

#endregion

#region Bayesian Average Rating Score

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of Star Rating
import scipy.stats as st
import math


def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)  # Derece sayısı (örneğin: 1 yıldız, 2 yıldız... 5 yıldız)
    z = st.norm.ppf(1 - (1 - confidence) / 2)  # Z değeri hesaplama
    N = sum(n)  # Toplam puan sayısı
    first_part = 0.0
    second_part = 0.0

    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n_k + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n_k + 1) / (N + K)

    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1 )

df.sort_values(by="weighted_sorting_score", ascending=False).head(20)
df.sort_values(by="bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5,1])].sort_values(by="bar_score", ascending=False).head(20)

#endregion

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - BAR Score

# Sorting Products
# - Sorting by Ratings
# - Sorting by Comment or Purchase
# - Sorting by Rating, Comment and Purchase
# - Sorting by BAR Score
# - Hybrid Sorting: BAR and others


#region Hybrid Sorting: BAR and others

def hybrid_rating_score(dataframe, bar_w =60, wss_w = 40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                    "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1 )
    wss_score = weighted_sorting_score(dataframe)

    return bar_score * bar_w/100 + wss_score * wss_w/100

df["hybrid_rating_score"] = hybrid_rating_score(df)
df.sort_values(by="hybrid_rating_score", ascending=False).head(20)
df[df["course_name"].str.contains("Veri Bilimi")].sort_values(by="hybrid_rating_score", ascending=False).head(20)
#endregion



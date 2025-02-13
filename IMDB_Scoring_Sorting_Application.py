import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets\movies_metadata.csv", low_memory=False)

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

#region Sorting by Vote Average

df.sort_values(by="vote_average", ascending=False).head(20)
df.sort_values(by="vote_average", ascending=False).head(20)
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70,0.80, 0.90, 0.95, 0.99])
df[df["vote_count"] > 400].sort_values(by="vote_average", ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values(by="average_count_score", ascending=False).head(20)
#endregion

#region IMDB Weighted Rating

# Weighted Rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the TOP 250
# C = the mean vote across the whole report (currently 7.0)

# Movie 1
# r = 8
# M = 500
# v = 1000
"""(v/(v+M) * r) = 5.33"""
"""(M/(v+M) * C) = 2.33"""
"""Total = 7.66"""

# Movie 2
# r = 8
# M = 500
# v = 3000
"""(v/(v+M) * r) = 6.85"""
"""(M/(v+M) * C) = 1"""
"""Total = 7.85"""



M = 2500
C = df["vote_average"].mean()

def weighted_rating(r,v,M,C):
    return ((v/(v+M) * r) + (M/(v+M) * C))

df.sort_values(by="average_count_score", ascending=False).head(20)

weighted_rating(7.40000,11444.000, M, C)
weighted_rating(8.1,14075.000, M, C)
weighted_rating(8.5,8358.000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

df.sort_values(by="weighted_rating", ascending=False).head(20)
#endregion

#region BAR Score
#The Dark Knight
#The Shawshank Redemption
#Fight Club
#Inception
#Pulp Fiction

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

bayesian_average_rating([34733, 4355, 4704,6561, 13515, 26183, 87368, 273082, 600260, 1295351])
#9.14538444560111

bayesian_average_rating()

df = pd.read_csv("datasets\imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"]= df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]]), axis=1 )
df.sort_values(by="bar_score", ascending=False).head(20)
#endregion


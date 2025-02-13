"""
#################################
SORTING REVIEWS
#################################
"""

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#region Up-Down Diff Score

# Review 1: 600 up 400 down
# Review 2: 5500 up 4500 down

def score_up_down_diff(up, down):
    return up - down

# Review 1
score_up_down_diff(600, 400)

# Review 2
score_up_down_diff(5500, 4500)

# Score = Avg. Score = up ratings / all ratings

def score_average_rating(up, down):
    if up - down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up 0 down
# Review 2: 100 up 1 down

score_average_rating(2, 0)
score_average_rating(100, 1)

#endregion

#region Wilson Lower Bound

def wilson_lower_bound(up,down,confidence=0.95):
    n = up + down
    if n == 0:
        return 0

    z= st.norm.ppf(1- (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600,400)
wilson_lower_bound(5500,4500)
wilson_lower_bound(2,0)
wilson_lower_bound(100,1)

#endregion

#region Case Study

up = [15,70,14,4,2,5,8,37,21,52,28,147,61,30,23,40,37,61,54,18,12,68]
down = [0,2,2,2,15,2,6,5,23,8,12,2,1,1,5,1,2,6,2,0,2,2]
comments = pd.DataFrame({"up":up, "down":down})

comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis=1)

comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

comments.sort_values(by="wilson_lower_bound", ascending=False)

comments_index = comments.sort_values(by="wilson_lower_bound", ascending=False).reset_index()
#endregion
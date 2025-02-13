"""
#################################
AB TESTING
##################################
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import scipy.stats as st
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import f_oneway
from scipy.stats import kruskal
from statsmodels.stats.proportion import proportions_ztest

#region Sampling

population = np.random.randint(0, 80 , 10000)
population.mean()

np.random.seed(115)

sample = np.random.choice(a = population, size = 100)
sample.mean()

np.random.seed(100)
sample1 = np.random.choice(a = population, size = 100)
sample2 = np.random.choice(a = population, size = 100)
sample3 = np.random.choice(a = population, size = 100)
sample4 = np.random.choice(a = population, size = 100)
sample5 = np.random.choice(a = population, size = 100)
sample6 = np.random.choice(a = population, size = 100)
sample7 = np.random.choice(a = population, size = 100)
sample8 = np.random.choice(a = population, size = 100)
sample9 = np.random.choice(a = population, size = 100)
sample10 = np.random.choice(a = population, size = 100)


(sample1.mean() + sample2.mean() + sample3.mean()  + sample4.mean()  + sample8.mean()  + sample7.mean()  + sample6.mean()  + sample5.mean()  + sample9.mean()  + sample10.mean()) / 10
#endregion

#region Descriptive Statistics

df = sns.load_dataset("tips")
df.head()
df.describe().T

#endregion

#region Confidence Intervals

df = sns.load_dataset("tips")
df.describe().T

df.head()

sms.DescrStatsW(df["total_bill"]).tconfint_mean()

# 18.66333170435847 - 20.908553541543164

df = sns.load_dataset("titanic")
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

#endregion

#region Correlation

df = sns.load_dataset("tips")
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter(x="tip", y="total_bill")
plt.show()

df["tip"].corr(df["total_bill"])
#endregion

#region AB Testing

# 1. Set Hypothesis
# 2. Control of Assumption
# - 1. Normality Assumption
# - 2. Homogeneity of Variance
# 3. Application of Hypothesis
# - 1. Mannwhineyu test if assumptions are not met
# - 2 . Independent two sample test if assumptions are met


#region Application 1: Smokers and NonSmokers


df = sns.load_dataset("tips")
df.head()

df["smoker"].value_counts()

df.groupby("smoker").agg({"total_bill": "mean"})

"""
smoker            
Yes      20.756344
No       19.188278

# Set Hypothesis
H0: M1 = M2
H1: M1 != M2

# Control of Assumption
1. Normality Assumption
2. Homogeneity of Variance

H0: Normal distribution assumption is met
H1: Normal distribution assumption is not met
"""

test_Stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_Stat, pvalue))

# Test Stat = 0.9045, pvalue = 0.0000
# p-value < 0.05, H0 IS WRONG!


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

# Test Stat = 0.9045, pvalue = 0.0000
# p-value < 0.05, H0 IS WRONG!

######################################
# Assumption of Homogenity of Variance
######################################

"""
H0: Variances are homogen.
H1: Variances are not homogen.
"""

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

# p-value < 0.05, H0 IS WRONG!

###########################
# Application of Hypthesis
##########################

# Act like assumptions are met.
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=False)
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

# Test Stat = 1.2843, pvalue = 0.2008
# p-value > 0.05, H0 CANNOT BE IGNORE!


# Assumptions are not met.
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

# Test Stat = 7531.5000, pvalue = 0.3413
# p-value > 0.05, H0 CANNOT BE IGNORE!

#endregion

#region Application 2: Male and Female Passengers
df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})

# 1. Set hypothesis.
# H0 = M1 = M2 ("There is no meanful difference between age average of Male and Female passengers.")
# H1 = M1 != M2 (There is.)

# 2. Control the assumptions

# Normality
# H0: Normal distribution assumption is met
# H1: Normal distribution assumption is not met

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# pvalue = 0.0071, H0 REJECTED

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# pvalue = 0.0000, H0 REJECTED

# Homogeneity of Variances
# H0: Variances are homogene.
# H1: Variances are not homogene.

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# pvalue = 0.0000, H0 CANNOT BE REJECTED

## NonParametric Test, because of assumptions cannot be met.

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# pvalue = 0.0261, H0 REJECTED







#endregion

#region Application 3: Differences of Age of Diaebetes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 0, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

#endregion

#region Application: Business Problem, Is there a statistical difference between the scores of those who watched the majority of the course and those who did not?

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df.groupby("Progress").agg({"Rating": "mean"})
df[df["Progress"] < 10]["Rating"].mean()
df[df["Progress"] > 70]["Rating"].mean()
df[df["Progress"] > 95]["Rating"].mean()

test_stat, pvalue = shapiro(df.loc[df["Progress"] < 25, "Rating"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df.loc[df["Progress"] < 25, "Rating"].dropna(),
                                 df.loc[df["Progress"] > 75, "Rating"].dropna())
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
#endregion


#endregion

#region AB Testing: Two-Sample Proportion Test

#H0: There is no meanful difference between to design.
#H1: There is.

succesfull_total = np.array([300,250])
view_total = np.array([1000,1100])

proportions_ztest(
    count=succesfull_total,
    nobs=view_total)

succesfull_total / view_total


#region Application: Is there a difference between survived ratio of males and females?

#H0: There isn't.
#H1: There is.

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "male", "survived"].mean()
df.loc[df["sex"] == "female", "survived"].mean()

male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()
female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs= [df.loc[df["sex"] == "female", "survived"].shape[0],df.loc[df["sex"] == "male", "survived"].shape[0]
                                             ])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
#endregion

#endregion

#region AB Testing: Analysis of Variance

# Used to compare more than two group means.

df = sns.load_dataset("tips")
df.groupby("day")["total_bill"].mean()


# H0: M1 = M2 = M3 = M4

# Control of Assumption

# Normality Assumption
# Homogeneity of Variance Assumption

# H0: Normality
for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"].dropna())[1]
    print("Group %s: pvalue = %.4f" % (group, pvalue))

# H0: Homogeneity of Variance

test_stat, pvalue = levene(df.loc[df["day"] == "Thur", "total_bill"].dropna(),
                           df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))


# Test of Hyptothesis

df.groupby("day").agg({"total_bill": ["mean", "median"]})

# parametric anova:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# nonparametric anova:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])


from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

tukey = comparison.tukeyhsd(0.10)
print(tukey.summary())
#endregion

#endregion





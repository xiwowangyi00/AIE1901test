import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("sample_tweets.csv")

# 1. Basic exploration
print(df.head())
print(df.describe())

# 2. Visualization
sns.boxplot(x="partisan_cue", y="anger_score", data=df)
plt.title("Anger score by presence of partisan cue")
plt.show()

# 3. Correlation
print("Correlation anger vs support:", df["anger_score"].corr(df["supports_policy"]))

# 4. Difference in means (t-test)
group1 = df[df["partisan_cue"]==1]["anger_score"]
group2 = df[df["partisan_cue"]==0]["anger_score"]
tstat, pval = stats.ttest_ind(group1, group2)
print("t =", tstat, "p =", pval)

# 5. OLS regression
ols = smf.ols("anger_score ~ partisan_cue", data=df).fit()
print(ols.summary())

# 6. Logistic regression
logit = smf.logit("supports_policy ~ anger_score + partisan_cue", data=df).fit()
print(logit.summary())

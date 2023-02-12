import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest
import statsmodels.api as sm
from scipy.stats import kstest
from scipy.stats import mannwhitneyu
import numpy as np 

# Load the data
df = pd.read_excel("assignment/assignment2/privacy 1.xlsx", header=0)


def analyze_distribution(df, column):
    #drop missing values
    column = df[column].dropna()
    
    #create histogram
    plt.hist(column, edgecolor='black', bins=5)
    plt.show()
    
    #create Q-Q plot with 45-degree line added to plot
    sm.qqplot(column, line='45')
    plt.show()
    
    #perform Shapiro-Wilk test for normality
    shapiro_results = shapiro(column)
    print("Shapiro-Wilk test results:", shapiro_results)
    
    #perform Kolmogorov-Smirnov test for normality
    kstest_results = kstest(column, 'norm')
    print("Kolmogorov-Smirnov test results:", kstest_results)

analyze_distribution(df, 'like_know')

#perform Mann-Whitney U test

group1 = df.query('classic_coke == 0')['like_know'].dropna().values
group2 = df.query('classic_coke == 1')['like_know'].dropna().values

u, p = mannwhitneyu(group1, group2)
print("Mann-Whitney U test results: u-statistic = {:.3f}, p-value = {:.10f}".format(u, p))


import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_excel("assignment/assignment2/privacy 1.xlsx", header=0)

# Create a subplot with two violin plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.violinplot(x="classic_coke", y="like_know", data=df, ax=ax[0])
sns.violinplot(x="gender", y="like_know", data=df, ax=ax[1])

# Add a title and labels to the plot
plt.suptitle("Violin Plots")
ax[0].set_xlabel("classic_coke")
ax[0].set_ylabel("like_know")
ax[1].set_xlabel("gender")
ax[1].set_ylabel("like_know")

# Show the plot
plt.show()

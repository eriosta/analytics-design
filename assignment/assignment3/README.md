# Assignment 3: Survival Analysis

## Quick Links
Go to [source code](https://github.com/eriosta/analytics-design/blob/main/assignment/assignment3/survival.py) to explore the `SurvivalTable` and `AdvancedSurvivalAnalysis` classes.

Go to [Jupyter Notebook](https://github.com/eriosta/analytics-design/blob/main/assignment/assignment3/assignment3.ipynb) for implementation and results.

## Tasks
1. Construct a survival table and plot a survival curve for a dataset of patients with Krusty the Clown disease.
2. For a dataset of patients with La Traviata disease, perform the following:
   1. Draw a survival plot showing survival curves for three different drugs.
   2. Test for an overall effect of any of the drugs on survival.
   3. Compare survival curves for each drug with each other and adjust for multiple group comparisons.

## Approach
The code here defines two classes, `SurvivalTable` and `AdvancedSurvivalAnalysis`.

The `SurvivalTable` class takes in survival data in the form of a dictionary or pandas DataFrame and creates a survival table from it. The `build_table` method is used to build the survival table by calculating the number of patients at risk, the number of deaths, the number of patients who survived past a certain time, and the probability of survival and death. The `print_table` method prints the survival table to the console, and the `plot_survival_curve` and `median_survival_time` methods plot the survival curve using the Kaplan-Meier estimator and calculate the median survival time, respectively.

The `AdvancedSurvivalAnalysis` class is used to perform advanced survival analysis on a dataset. The `plot_survival_curves` method draws a survival plot that shows the survival curves for all groups. The `test_global_effect` method tests to see if overall there is an effect of any of the drugs on survival taken as a global set. The `test_pairwise_effects` method compares the survival curves for each group and sees if any of the curves are different from each other.

## Results
We analyzed a dataset of patients with La Traviata disease and investigated the effect of three different drugs on patient survival.

First, we plotted survival curves for each drug using the Kaplan-Meier method. The survival curves showed that patients on drug 2 had the highest survival rate, followed by patients on drug 3, and patients on drug 1 had the lowest survival rate.

Next, we performed a Log-Rank test to determine if there was an overall effect of any of the drugs on survival. The Log-Rank test revealed a p-value of 1.6240630634437227e-62, indicating a significant difference between at least one pair of survival curves.

To further investigate the pairwise differences between the drugs, we performed multiple pairwise comparisons using the Bonferroni correction to adjust for multiple group comparisons. The results showed that there was a significant difference in survival curves between drug 1 and drug 2, with a corrected p < 0.00001. There was also a significant difference in survival curves between drug 2 and drug 3, with a corrected p-value of 0.0002. However, there was no significant difference in survival curves between drug 1 and drug 3, with a corrected p-value of 0.3691.

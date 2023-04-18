import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import statsmodels.api as sm

class MarketResearch:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)

        # Drop rows containing non-digits
        self.df = self.df[~self.df['pizza'].str.contains(r'\D', na=False)]

        # Create contingency table
        self.contingency_table = pd.crosstab(self.df['RESP_RACE'], self.df['pizza'])

        # Generate Table object for chi-squared test and correspondence analysis
        self.table_obj = sm.stats.Table(self.contingency_table)

        # Get chi-squared test results
        self.chi2 = self.table_obj.test_nominal_association().statistic
        self.p_value = self.table_obj.test_nominal_association().pvalue
        self.dof = self.table_obj.test_nominal_association().df
        
        # Get expected frequencies
        self.expected = self.table_obj.fittedvalues
        
    def expected_frequencies(self):
        return pd.DataFrame(self.expected, columns=self.contingency_table.columns, index=self.contingency_table.index)

    def chi_square_contributions(self):
        chi_square_contributions = (self.contingency_table - self.expected) ** 2 / self.expected
        return chi_square_contributions


    def critical_value(self, significance_level=0.05):
        return chi2.ppf(1 - significance_level, self.dof)

    def reject_null_hypothesis(self, significance_level=0.05):
        return self.chi2 > self.critical_value(significance_level)

    def correspondence_analysis(self):
        row_mass = self.contingency_table.sum(axis=1) / self.contingency_table.sum().sum()
        col_mass = self.contingency_table.sum(axis=0) / self.contingency_table.sum().sum()
        inertia = (self.contingency_table / self.contingency_table.values.sum()).values.ravel()
        quality = (self.chi_square_contributions() / self.chi2).values.ravel()

        row_inertia = pd.DataFrame({'Mass': row_mass, 'Inertia': inertia[::len(col_mass)]})
        col_inertia = pd.DataFrame({'Mass': col_mass, 'Inertia': inertia[:len(col_mass)]})
        row_quality = pd.DataFrame({'Mass': row_mass, 'Quality': quality[::len(col_mass)]})
        col_quality = pd.DataFrame({'Mass': col_mass, 'Quality': quality[:len(col_mass)]})

        return {"Row Mass": row_mass,
                "Column Mass": col_mass,
                "Row Inertia": row_inertia,
                "Column Inertia": col_inertia,
                "Row Quality": row_quality,
                "Column Quality": col_quality}

    def standardized_adjusted_residuals(self):
        standardized_adjusted_residuals = (self.contingency_table - self.expected) / np.sqrt(self.expected)
        return standardized_adjusted_residuals
    
    def plot_raw_association(self):
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        # Create heatmap
        heatmap = ax.pcolor(self.contingency_table, cmap=plt.cm.Blues)

        # Add colorbar
        cbar = plt.colorbar(heatmap)

        # Set tick labels and axis labels
        ax.set_xticks(np.arange(self.contingency_table.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(self.contingency_table.shape[0])+0.5, minor=False)
        ax.set_xticklabels(['Pizza Hut', 'Domino\'s Pizza', 'Papa Johns', 'Little Caesars'], minor=False, rotation=45)
        ax.set_yticklabels(['White', 'Black', 'Asian', 'Other'], minor=False)

        # Print values in each cell
        for i in range(self.contingency_table.shape[0]):
            for j in range(self.contingency_table.shape[1]):
                ax.text(j+0.5, i+0.5, self.contingency_table.iloc[i, j], ha='center', va='center')

        # Set axis labels
        ax.set_xlabel('Pizza Chain')
        ax.set_ylabel('Race')

        plt.show()

    def plot_norm_association(self):
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        # Normalize contingency table
        normalized_table = self.contingency_table.div(self.contingency_table.sum(axis=1), axis=0)

        # Create heatmap
        heatmap = ax.pcolor(normalized_table, cmap=plt.cm.Blues)

        # Add colorbar
        cbar = plt.colorbar(heatmap)

        # Set tick labels and axis labels
        ax.set_xticks(np.arange(normalized_table.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(normalized_table.shape[0])+0.5, minor=False)
        ax.set_xticklabels(['Pizza Hut', 'Domino\'s Pizza', 'Papa Johns', 'Little Caesars'], minor=False, rotation=45)
        ax.set_yticklabels(['White', 'Black', 'Asian', 'Other'], minor=False)

        # Print values in each cell
        for i in range(normalized_table.shape[0]):
            for j in range(normalized_table.shape[1]):
                ax.text(j+0.5, i+0.5, '{:.2f}'.format(normalized_table.iloc[i, j]), ha='center', va='center')

        # Set axis labels
        ax.set_xlabel('Pizza Chain')
        ax.set_ylabel('Race')

        plt.show()


# initialize a MarketResearch object with a data file
# mr = MarketResearch('assignment/assignment4/pizzafem truncated.csv')
# mr.contingency_table.to_csv('data.csv')
# # calculate and print the expected frequencies of the contingency table
# print(mr.expected_frequencies())

# # calculate and print the chi-square contributions of the contingency table
# print(mr.chi_square_contributions())

# # calculate and print the critical value for rejecting the null hypothesis
# print(mr.critical_value())

# # determine and print whether the null hypothesis should be rejected
# print(mr.reject_null_hypothesis())

# # perform correspondence analysis and return row and column dataframes, coordinates, and explained variance ratios
# tables = mr.correspondence_analysis()

# print("Row Mass:\n", tables["Row Mass"])
# print("Column Mass:\n", tables["Column Mass"])
# print("Row Inertia:\n", tables["Row Inertia"])
# print("Column Inertia:\n", tables["Column Inertia"])
# print("Row Quality:\n", tables["Row Quality"])
# print("Column Quality:\n", tables["Column Quality"])

# # calculate and print the standardized adjusted residuals of the contingency table
# print(mr.standardized_adjusted_residuals())

# # mr.plot_raw_association()

# mr.plot_norm_association()

# # mr.contingency_table



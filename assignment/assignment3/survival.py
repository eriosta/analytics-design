import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

data1 = {
    'subject': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'time': [1, 4, 5, 5, 7, 10, 10, 12, 14, 16],
    'event': [1, 1, 0, 1, 1, 0, 0, 1, 1, 0]
}

data1 = pd.DataFrame(data1)

data2 = [
    [1, 681, 0],
    [1, 602, 0],
    [1, 996, 0],
    [1, 1162, 0],
    [1, 833, 0],
    [1, 477, 0],
    [1, 630, 0],
    [1, 596, 0],
    [1, 226, 0],
    [1, 699, 0],
    [1, 811, 0],
    [1, 530, 0],
    [1, 482, 0],
    [1, 367, 0],
    [1, 118, 1],
    [1, 83, 1],
    [1, 76, 1],
    [1, 104, 1],
    [1, 109, 1],
    [1, 72, 1],
    [1, 87, 1],
    [1, 162, 1],
    [1, 94, 1],
    [1, 30, 1],
    [1, 26, 1],
    [1, 22, 1],
    [1, 49, 1],
    [1, 74, 1],
    [1, 122, 1],
    [1, 86, 1],
    [1, 66, 1],
    [1, 92, 1],
    [1, 109, 1],
    [1, 255, 1],
    [1, 1, 1],
    [1, 107, 1],
    [1, 110, 1],
    [1, 232, 1],
    [2, 2569, 0],
    [2, 2506, 0],
    [2, 2409, 0],
    [2, 2218, 0],
    [2, 1857, 0],
    [2, 1829, 0],
    [2, 1562, 0],
    [2, 1470, 0],
    [2, 1363, 0],
    [2, 1030, 0],
    [2, 1860, 0],
    [2, 1258, 0],
    [2, 2246, 0],
    [2, 1870, 0],
    [2, 1799, 0],
    [2, 1709, 0],
    [2, 1674, 0],
    [2, 1568, 0],
    [2, 1527, 0],
    [2, 1324, 0],
    [2, 1957, 0],
    [2, 1932, 0],
    [2, 1847, 0],
    [2, 1848, 0],
    [2, 1850, 0],
    [2, 1843, 0],
    [2, 1535, 0],
    [2, 1447, 0],
    [2, 1384, 0],
    [2, 914, 1],
    [2, 2204, 1],
    [2, 1063, 1],
    [2, 481, 1],
    [2, 605, 1],
    [2, 641, 1],
    [2, 390, 1],
    [2, 288, 1],
    [2, 421, 1],
    [2, 1379, 1],
    [2, 1748, 1],
    [2, 486, 1],
    [2, 448, 1],
    [2, 272, 1],
    [2, 1074, 1],
    [2, 1381, 1],
    [2, 1410, 1],
    [2, 1353, 1],
    [2, 1480, 1],
    [2, 435, 1],
    [2, 248, 1],
    [2, 1704, 1],
    [2, 1411, 1],
    [2, 219, 1],
    [2, 606, 1],
    [3, 2640, 0],
    [3, 2430, 0],
    [3, 2252, 0],
    [3, 2140, 0],
    [3, 2133, 0],
    [3, 1738, 0],
    [3, 2631, 0],
    [3, 2524, 0],
    [3, 1845, 0],
    [3, 1936, 0],
    [3, 1845, 0],
    [3, 422, 1],
    [3, 162, 1],
    [3, 84, 1],
    [3, 100, 1],
    [3, 212, 1],
    [3, 47, 1],
    [3, 242, 1],
    [3, 456, 1],
    [3, 268, 1],
    [3, 318, 1],
    [3, 732, 1],
    [3, 467, 1],
    [3, 947, 1],
    [3, 390, 1],
    [3, 183, 1],
    [3, 105, 1],
    [3, 115, 1],
    [3, 164, 1],
    [3, 693, 1],
    [3, 120, 1],
    [3, 80, 1],
    [3, 677, 1],
    [3, 64, 1],
    [3, 168, 1],
    [3, 874, 1],
    [3, 616, 1],
    [3, 157, 1],
    [3, 625, 1],
    [3, 48, 1],
    [3, 273, 1],
    [3, 163, 1],
    [3, 376, 1],
    [3, 113, 1]]

data2 = pd.DataFrame(data2, columns=['group', 'time', 'event'])

def confirm_import(func):
    def wrapper(*args, **kwargs):
        print("Classes successfully imported.")
        return func(*args, **kwargs)
    return wrapper

@confirm_import
class SurvivalTable:
    """
    A class to construct and print a survival table from a pandas DataFrame
    containing survival data.
    """
    def __init__(self, data):
        """
        Initialize the SurvivalTable object with the given data.

        Parameters:
        -----------
        data : dict or pandas.DataFrame
            The survival data as a dictionary or pandas DataFrame.
            It should have columns for 'subject', 'time', and 'event'.

        """
        self.df = pd.DataFrame(data).sort_values(by='time')
        self.survival_table = pd.DataFrame(columns=['N', 'D', 'S', 'S_prob', 'D_prob'])

    def build_table(self):
        """
        Build the survival table from the initialized data.

        Returns:
        --------
        pandas.DataFrame
            The survival table as a pandas DataFrame.

        """
        # Set the initial number of patients at risk (N) to the total number of patients
        N = len(self.df)

        # Loop through each unique time point
        for t in self.df['time'].unique():
            # Get the subset of the data for patients who have a time value greater than or equal to t
            subset = self.df[self.df['time'] >= t]

            # Get the number of patients who have a time value greater than or equal to t
            n = len(subset)

            # Get the number of deaths that occurred at time t
            d = sum(subset['event'])

            # Get the number of patients who survived past time t
            s = n - d

            # Calculate the probability of survival and the probability of death
            s_prob = s / N
            d_prob = d / N

            # Add the values to the survival table
            self.survival_table.loc[t, :] = [N, d, s, s_prob, d_prob]

            # Update the number of patients at risk (N) for the next time point
            N = n

        # Set the index name to 'Time'
        self.survival_table.index.name = 'Time'

    def print_table(self):
        """
        Print the survival table to the console.

        """
        print(self.survival_table)

    def plot_survival_curve(self):
        """
        Plot the survival curve using the Kaplan-Meier estimator.

        """
        kmf = KaplanMeierFitter()
        kmf.fit(self.df['time'], self.df['event'])
        kmf.plot()
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Survival Curve')
        plt.show()

    def median_survival_time(self):
        """
        Calculate the median survival time using the Kaplan-Meier estimator.

        Returns:
        --------
        float
            The median survival time.
        """
        kmf = KaplanMeierFitter()
        kmf.fit(self.df['time'], self.df['event'])
        return kmf.median_survival_time_

@confirm_import
class AdvancedSurvivalAnalysis:
    """
    A class for performing advanced survival analysis on a dataset.
    
    Parameters
    ----------
    data : dict or pandas.DataFrame
        The survival data as a dictionary or pandas DataFrame.
        It should have columns for 'group', 'time', and 'event'.
    """

    def __init__(self, data):
        """
        Initializes the class with the given data.
        """
        self.df = pd.DataFrame(data)
        self.groups = self.df['group'].unique()

    def plot_survival_curves(self):
        """
        Draws a survival plot that shows the survival curves for all groups.
        """
        for g in self.groups:
            subset = self.df[self.df['group'] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(subset['time'], subset['event'], label=f'Group {g}')
            kmf.plot()

        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Survival Curves by Group')
        plt.show()

    def test_global_effect(self, alpha=0.05):
        """
        Tests to see if overall there is an effect of any of the drugs on survival taken as a global set.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the test. Default is 0.05.

        Returns
        -------
        bool
            True if there is a significant difference between at least one pair of survival curves, False otherwise.
        """
        results = logrank_test(self.df['time'], self.df['group'], self.df['event'])
        print('Log-Rank Test')
        print('p-value:', results.p_value)
        if results.p_value < alpha / (len(self.groups) * (len(self.groups) - 1) / 2):
            print('There is a significant difference between at least one pair of survival curves.')
            return True
        else:
            print('There is no significant difference between any pair of survival curves.')
            return False

    def test_pairwise_effects(self, alpha=0.05, correction='bonferroni'):
        """
        Compares the survival curves for each group and sees if any of the curves are different from each other.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the test. Default is 0.05.
        correction : str, optional
            The type of multiple testing correction to apply. Can be 'bonferroni', 'holm-sidak', 'holm',
            'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', or None. Default is 'bonferroni'.

        Returns
        -------
        None
        """
        pairs = list(combinations(self.groups, 2))
        p_values = []
        test_statistics = []
        for pair in pairs:
            subset1 = self.df[self.df['group'] == pair[0]]
            subset2 = self.df[self.df['group'] == pair[1]]
            results = logrank_test(subset1['time'], subset2['time'], subset1['event'], subset2['event'])
            p_values.append(results.p_value)
            test_statistics.append(results.test_statistic)
            print(f'Group {pair[0]} vs Group {pair[1]}: p-value = {results.p_value:.4f}, test statistic = {results.test_statistic:.2f}')
        # Perform multiple testing correction
        if correction is not None:
            rejected, p_values_corrected, _, _ = multipletests(p_values, alpha=alpha, method=correction)
            for i, pair in enumerate(pairs):
                if rejected[i]:
                    stat = test_statistics[i]
                    print(f'Group {pair[0]} and Group {pair[1]} have significantly different survival curves.')
                    # print(f'Logrank Test Statistic: {stat:.2f}')
        else:
            for i, pair in enumerate(pairs):
                if p_values[i] < alpha:
                    stat = test_statistics[i]
                    print(f'Group {pair[0]} and Group {pair[1]} have significantly different survival curves.')
                    # print(f'Logrank Test Statistic: {stat:.2f}')



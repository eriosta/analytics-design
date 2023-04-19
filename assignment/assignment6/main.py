import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents

class BeerConsumptionForecast:
    """
    This class is designed to forecast beer consumption using the Unobserved Components Model (UCM) time series model.
    
    Usage example:
    ```python
    # Read data from a CSV file
    path = 'data.csv'
    beer_forecast = BeerConsumptionForecast(path)

    # Calculate adjusted R-squared
    beer_forecast.adjusted_r_squared()

    # Show component significance
    beer_forecast.component_significance()

    # Forecast the next 48 time periods
    beer_forecast.forecast()

    # Plot observed and forecasted data
    beer_forecast.plot_data()
    ```
    """
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, names=['time_period', 'consumption'], header=None)
        self.data.set_index('time_period', inplace=True)

    def fit_ucm(self):
        """
        Fit the Unobserved Components Model using the historical beer consumption data.
        """
        self.model = UnobservedComponents(self.data['consumption'], level='local linear trend', seasonal=48)
        self.results = self.model.fit()

    def adjusted_r_squared(self):
        """
        Calculate the adjusted R-squared value for the Unobserved Components Model.

        :return: A float representing the adjusted R-squared value.
        """
        self.fit_ucm()
        self.fitted_values = self.results.fittedvalues
        residuals = self.data['consumption'] - self.fitted_values
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.data['consumption'] - np.mean(self.data['consumption'])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        n = len(self.data)
        k = self.results.df_model + 1  # Adding 1 to account for the intercept
        adj_r2 = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))

        # print the full table with fitted_values and data['consumption']
        # set the display options to show all rows
        # pd.set_option('display.max_rows', None)
        # print(pd.concat([self.fitted_values, self.data['consumption']], axis=1))
        
        return adj_r2

    def component_significance(self):
        """
        Generate a summary table with information about the fitted model and the significance of its components.

        :return: A string containing the summary table.
        """
        return self.results.summary()

    def forecast(self, steps):
        """
        Forecast the next `steps` time periods of beer consumption data.

        :param steps: An integer representing the number of time periods to forecast (default: 48).
        :return: A pandas Series containing the forecasted beer consumption data.
        """
        self.forecasted_data = self.results.get_forecast(steps=steps).predicted_mean
        self.forecasted_data.index = pd.RangeIndex(start=self.data.index[-1] + 1, stop=self.data.index[-1] + steps + 1)
        return self.forecasted_data


    def plot_data(self, steps):
        """
        Plot the observed historical data and the forecasted data.

        :param steps: An integer representing the number of time periods to forecast (default: 48).
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data, label='Observed')
        plt.plot(self.forecast(steps), label='Forecasted', linestyle='--')
        plt.legend(loc='upper left')
        plt.xlabel('Time Period')
        plt.ylabel('Beer Consumption')
        plt.title('Beer Consumption Forecast')
        plt.show()

import numpy as np


class MultiQuantileBacktest:
    """
    Implements a multiquantile backtesting model for Expected Shortfall.
    This model evaluates multiple quantiles of the distribution of returns
    or losses to assess the accuracy of VaR and ES projections.

    Inputs and outputs are consistent with the EStestRidge class for easy integration.
    """

    def __init__(self, X_obs, X, VaRLevel, VaR, ES, nSim, alpha):
        """
        Initialize the MultiQuantileBacktest class with input parameters.

        :param X_obs: Array of observed portfolio returns.
        :type X_obs: np.array
        :param X: Function that simulates portfolio returns under the null hypothesis (H0).
        :type X: function
        :param VaRLevel: VaR confidence level (e.g., 0.05 for 95%).
        :type VaRLevel: float
        :param VaR: Array of projected Value at Risk (VaR) estimates for each period.
        :type VaR: np.array
        :param ES: Array of projected Expected Shortfall (ES) estimates for each period.
        :type ES: np.array
        :param nSim: Number of Monte Carlo simulations to perform.
        :type nSim: int
        :param alpha: Significance level for the test (e.g., 0.05).
        :type alpha: float
        """
        self.mean_breach_value = 0
        self.X_obs = X_obs
        self.T = X_obs.size
        self.X = X
        self.VaR = -VaR
        self.ES = -ES
        self.VaRLevel = VaRLevel
        self.nSim = nSim
        self.alpha = alpha
        self.quantile_levels = np.linspace(0, VaRLevel, 10)  # Set quantile levels for evaluation
        self.simulation()

    def statistic(self, X, I):
        """
        Calculate the multiquantile test statistic Z, which evaluates multiple quantiles.

        :param X: Simulated or observed returns.
        :type X: np.array
        :param I: Indicator array where each element is 1 if VaR is breached, otherwise 0.
        :type I: np.array
        :return: Test statistic Z as a float.
        :rtype: float
        """
        Z = 0
        for q in self.quantile_levels:
            quantile = np.percentile(X, 100 * q)
            Z += np.sum((X * I) / self.ES) / (self.VaRLevel * self.T)
        return Z / len(self.quantile_levels)

    def simulation(self):
        """
        Perform Monte Carlo simulations to calculate the critical value and p-value.

        The p-value is the proportion of simulated test statistics that are less than
        the test statistic calculated from observed returns.
        """
        I_obs = (self.X_obs + self.VaR < 0)  # Identify instances where VaR is breached
        self.VaR_breaches = I_obs.sum()
        self.Z_obs = self.statistic(self.X_obs, I_obs)

        # Calculate mean breach value if there are breaches
        if len(self.X_obs[I_obs]) > 0:
            self.mean_breach_value = np.mean(self.X_obs[I_obs])
        else:
            self.mean_breach_value = 0

        statistics = []
        for _ in range(self.nSim):
            X_i = self.X()
            I_i = (X_i + self.VaR < 0)
            Z_i = self.statistic(X_i, I_i)
            statistics.append(Z_i)

        self.critical_value = np.quantile(statistics, self.alpha)
        self.p_value = np.mean([1 if stat < self.Z_obs else 0 for stat in statistics])

    def print(self):
        """
        Prints a summary of the test results including statistics and significance levels.
        """
        print('----------------------------------------------------------------')
        print('        Multiquantile Expected Shortfall Test by Simulation      ')
        print('----------------------------------------------------------------')
        print('Number of observations: ' + str(self.T))
        print('Number of VaR breaches: ' + str(self.VaR_breaches))
        print('Expected number of VaR breaches: ' + str(self.T * self.VaRLevel))
        print('ES Statistic: ' + str(self.Z_obs))
        print('Expected ES Statistic under the null hypothesis: ' + str(0))
        print('Critical Value at α = ' + str(self.alpha) + ': ' + str(self.critical_value))
        print('p-value: ' + str(self.p_value))
        print('Number of Monte Carlo simulations: ' + str(self.nSim))
        print('----------------------------------------------------------------')

    def get_results_summary(self):
        """
        Returns a summary of the test results and relevant statistics as a formatted string.

        :return: Summary of test results.
        :rtype: str
        """
        result = []
        result.append('----------------------------------------------------------------')
        result.append('        Multiquantile Expected Shortfall Test by Simulation      ')
        result.append('----------------------------------------------------------------')
        result.append('Number of observations: ' + str(self.T))
        result.append('Number of VaR breaches: ' + str(self.VaR_breaches))
        result.append('Expected number of VaR breaches: ' + str(self.T * self.VaRLevel))
        result.append('ES Statistic: ' + str(self.Z_obs))
        result.append('Expected ES Statistic under the null hypothesis: ' + str(0))
        result.append('Critical Value at α = ' + str(self.alpha) + ': ' + str(self.critical_value))
        result.append('p-value: ' + str(self.p_value))
        result.append('Number of Monte Carlo simulations: ' + str(self.nSim))
        result.append('----------------------------------------------------------------')

        return '\n'.join(result)

    def backtest_out(self):
        """
        Outputs key results for further analysis.

        :return: Tuple containing the number of VaR breaches and the mean breach value.
        :rtype: tuple(int, float)
        """
        return self.VaR_breaches, self.mean_breach_value

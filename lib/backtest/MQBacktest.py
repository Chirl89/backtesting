import numpy as np

class MultiQuantileBacktest:
    """
    Implements a multiquantile backtesting model for Expected Shortfall.
    This model evaluates multiple quantiles of the distribution of returns
    or losses to assess the accuracy of VaR and ES projections.

    Inputs and outputs are kept identical to the EStestRidge class for consistency.
    """

    def __init__(self, X_obs, X, VaRLevel, VaR, ES, nSim, alpha):
        """
        Initialize the MultiQuantileBacktest class with inputs similar to EStestRidge.

        Parameters are identical in nature to those used in the EStestRidge class.
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
        self.quantile_levels = np.linspace(0, VaRLevel, 10)  # Example quantile levels
        self.simulation()

    def statistic(self, X, I):
        """
        Calculate the multiquantile statistic Z, which compares multiple quantiles.
        """
        Z = 0
        for q in self.quantile_levels:
            quantile = np.percentile(X, 100 * q)
            Z += np.sum((X * I) / self.ES) / (self.VaRLevel * self.T)
        return Z / len(self.quantile_levels)

    def simulation(self):
        """
        Perform Monte Carlo simulations to obtain the critical value and p-value.
        """
        I_obs = (self.X_obs + self.VaR < 0)
        self.VaR_breaches = I_obs.sum()
        self.Z_obs = self.statistic(self.X_obs, I_obs)

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
        Prints the results of the test and other useful information.
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
        Returns a summary of the test results and other useful information as a string.
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
        """
        return self.VaR_breaches, self.mean_breach_value

import numpy as np


class FisslerZiegelBacktest:
    """
    Implements the Fissler-Ziegel backtesting model for Expected Shortfall.
    This test evaluates the joint accuracy of VaR and ES using scoring functions.

    Reference:
    Fissler, Tobias, and Johanna F. Ziegel. "Higher order elicitability and Osband's principle."
    The Annals of Statistics 44.4 (2016): 1680-1707.
    """

    def __init__(self, X_obs, X, VaRLevel, VaR, ES, nSim, alpha):
        """
        Initialize the FisslerZiegelBacktest class with input parameters.

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
        self.VaR_breaches = 0
        self.X_obs = X_obs
        self.T = X_obs.size
        self.X = X
        self.VaR = -VaR  # Convert to positive values if given as negative (losses)
        self.ES = -ES  # Convert to positive values if given as negative (losses)
        self.VaRLevel = VaRLevel
        self.nSim = nSim
        self.alpha = alpha

        self.simulation()

    def scoring_function(self, x, VaR, ES):
        """
        Fissler-Ziegel scoring function for joint evaluation of VaR and ES.

        :param x: Observed returns.
        :type x: np.array
        :param VaR: Projected Value at Risk (VaR) estimate.
        :type VaR: np.array
        :param ES: Projected Expected Shortfall (ES) estimate.
        :type ES: np.array
        :return: Scoring function values for each observation.
        :rtype: np.array
        """
        indicator = (x <= VaR).astype(float)
        score = ((1 / ES) * (VaR - x) * (indicator - self.VaRLevel) + (1 - VaR / ES))
        return score

    def statistic(self, X):
        """
        Calculate the Fissler-Ziegel test statistic as the mean of the scoring function.

        :param X: Simulated or observed returns.
        :type X: np.array
        :return: The test statistic (mean of scoring function).
        :rtype: float
        """
        scores = self.scoring_function(X, self.VaR, self.ES)
        return np.mean(scores)

    def simulation(self):
        """
        Perform Monte Carlo simulations to obtain the critical value and p-value.

        The p-value is the fraction of scenarios where the simulated test statistic
        is smaller than the observed test statistic.
        """
        self.Z_obs = self.statistic(self.X_obs)
        I_obs = (self.X_obs + self.VaR < 0)
        self.VaR_breaches = I_obs.sum()
        if len(self.X_obs[I_obs]) > 0:
            self.mean_breach_value = np.mean(self.X_obs[I_obs])
        else:
            self.mean_breach_value = 0

        statistics = []
        for _ in range(self.nSim):
            X_i = self.X()
            Z_i = self.statistic(X_i)
            statistics.append(Z_i)

        self.critical_value = np.quantile(statistics, self.alpha)
        self.p_value = np.mean([1 if stat < self.Z_obs else 0 for stat in statistics])

    def print(self):
        """
        Prints a summary of the Fissler-Ziegel test results.
        """
        print('----------------------------------------------------------------')
        print('          Fissler-Ziegel Expected Shortfall Test by Simulation   ')
        print('----------------------------------------------------------------')
        print('Number of observations: ' + str(self.T))
        print('ES Statistic: ' + str(self.Z_obs))
        print('Expected ES Statistic under the null hypothesis: ' + str(0))
        print('Critical Value at α = ' + str(self.alpha) + ': ' + str(self.critical_value))
        print('p-value: ' + str(self.p_value))
        print('Number of Monte Carlo simulations: ' + str(self.nSim))
        print('----------------------------------------------------------------')

    def get_results_summary(self):
        """
        Returns a summary of the Fissler-Ziegel test results as a string.

        :return: Summary of test results.
        :rtype: str
        """
        result = []
        result.append('----------------------------------------------------------------')
        result.append('          Fissler-Ziegel Expected Shortfall Test by Simulation   ')
        result.append('----------------------------------------------------------------')
        result.append('Number of observations: ' + str(self.T))
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

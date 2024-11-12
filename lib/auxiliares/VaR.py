import numpy as np


def var_vol(volatilidad, confidence_level):
    """
    Calculates the Value at Risk (VaR) based on provided volatility and confidence level.

    :param volatilidad: A DataFrame or Series containing a 'VOLATILITY' column with volatility values.
    :type volatilidad: pandas.DataFrame or pandas.Series
    :param confidence_level: The confidence level for the VaR calculation (e.g., 0.95 for 95% confidence).
    :type confidence_level: float
    :return: Daily Value at Risk (VaR) values as an array, representing the loss threshold at the specified confidence level.
    :rtype: numpy.ndarray
    """

    # Calculate the z-score for the specified confidence level using a normal distribution
    z_score = np.percentile(np.random.normal(0, 1, 100000), (1 - confidence_level) * 100)

    # Calculate the daily VaR by scaling the z-score by the volatility values
    var_diario = z_score * volatilidad['VOLATILITY'].values

    # Return the calculated daily VaR values
    return var_diario

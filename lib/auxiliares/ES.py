import numpy as np


def expected_shortfall(volatilidad, nivel_confianza):
    """
    Calculates the Expected Shortfall (ES), a risk measure indicating the expected loss in worst-case scenarios.

    :param volatilidad: A DataFrame or Series containing a 'VOLATILITY' column with volatility values.
    :type volatilidad: pandas.DataFrame or pandas.Series
    :param nivel_confianza: The confidence level for the ES calculation (e.g., 0.95 for 95% confidence).
    :type nivel_confianza: float
    :return: The expected shortfall as a negative value, representing expected loss at the specified confidence level.
    :rtype: numpy.ndarray
    """

    # Calculate the z-score corresponding to the specified confidence level
    z_score = np.percentile(np.random.normal(0, 1, 100000), (1 - nivel_confianza) * 100)

    # Compute the probability density function (PDF) value at the z-score
    phi_z_score = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z_score ** 2)

    # Calculate the Expected Shortfall by adjusting volatility values according to the confidence level and PDF
    es = (volatilidad['VOLATILITY'].values / (1 - nivel_confianza)) * phi_z_score

    # Return the ES as a negative value to indicate the expected loss
    return -es

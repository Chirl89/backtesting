import numpy as np


def es_real(data, confidence_level, start_date, end_date):
    """
    Calculates the real Expected Shortfall (ES) and the number of exceptions over a specified period.

    :param data: A DataFrame containing 'Log Returns' for each date.
    :type data: pandas.DataFrame
    :param confidence_level: Confidence level for calculating the Value at Risk (VaR) and Expected Shortfall (e.g., 0.975).
    :type confidence_level: float
    :param start_date: Start date for the analysis period.
    :type start_date: str or datetime
    :param end_date: End date for the analysis period.
    :type end_date: str or datetime
    :return: A tuple containing the Expected Shortfall (ES) and the count of exceptions.
    :rtype: tuple(float, int)
    """

    # Filter data to the specified date range
    data = data[start_date:end_date].copy()

    # Calculate the Value at Risk (VaR) at the specified confidence level
    var = np.percentile(data['Log Returns'].dropna(), (1 - confidence_level) * 100)

    # Identify exceptions (returns that fall below the VaR threshold)
    if data[data['Log Returns'] < var].empty:
        exceptions = 0
    else:
        exceptions = data[data['Log Returns'] < var]

    # Calculate the Expected Shortfall as the mean of the exceptions
    if data[data['Log Returns'] < var].empty:
        es = 0
    else:
        es = exceptions['Log Returns'].mean()

    # Count the number of exceptions
    try:
        exceptions_count = len(exceptions)
    except:
        exceptions_count = 0

    # Return the Expected Shortfall and the count of exceptions
    return es, exceptions_count

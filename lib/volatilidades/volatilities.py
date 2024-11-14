import numpy as np
from arch import arch_model


def calculate_std_volatility(data, window=100):
    """
    Calculate the standard deviation-based rolling volatility over a specified window.

    :param data: DataFrame containing at least a 'Log Returns' column.
    :param window: The window size for calculating the rolling standard deviation.
    :return: Series with the rolling standard deviation, representing volatility.
    """
    returns = data['Log Returns'].dropna()  # Drop missing values in returns data
    return returns.rolling(window=window).std().dropna()  # Calculate rolling std deviation


def calculate_ewma_volatility(data, lambda_=0.94):
    """
    Calculate the Exponentially Weighted Moving Average (EWMA) volatility.

    :param data: DataFrame containing at least a 'Log Returns' column.
    :param lambda_: Decay factor for EWMA, commonly set to 0.94 in finance.
    :return: Series of EWMA volatility, calculated as the square root of EWMA variance.
    """
    returns = data['Log Returns'].dropna()  # Drop missing values in returns data
    ewma_variance = returns ** 2  # Initialize variance with squared returns

    # Calculate EWMA variance for each time step
    for t in returns.index[1:]:
        ewma_variance[t] = lambda_ * ewma_variance.shift(1)[t] + (1 - lambda_) * returns[t] ** 2

    current_volatility = np.sqrt(ewma_variance)  # Take the square root to get volatility

    return current_volatility


def calculate_gjrgarch_volatility(data, scale_factor=100):
    """
    Calculate volatility using the GJR-GARCH(1,1,1) model, which accounts for asymmetry in returns.

    :param data: DataFrame containing at least a 'Log Returns' column.
    :param scale_factor: Factor to scale returns to avoid numerical issues in estimation.
    :return: Series of conditional volatility based on the GJR-GARCH model.
    """
    returns = data['Log Returns'].dropna()  # Drop missing values in returns data
    returns_scaled = returns * scale_factor  # Scale returns to improve model stability

    # Define and fit the GJR-GARCH model
    model = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, rescale=False)
    model_fit = model.fit(disp="off")  # Suppress output during fitting
    conditional_volatility = model_fit.conditional_volatility

    # Scale back the volatility to the original scale
    conditional_volatility = conditional_volatility / scale_factor

    return conditional_volatility


def calculate_volatilities(data):
    """
    Calculate various volatility measures (EWMA and GJR-GARCH) for a dataset.

    :param data: DataFrame containing at least a 'Log Returns' column.
    :return: Dictionary with different types of volatility measures as Series.
    """
    volatilities = {
        'STD': calculate_std_volatility(data),  # Uncomment to include standard deviation volatility
        'EWMA': calculate_ewma_volatility(data),
        'GJR_GARCH': calculate_gjrgarch_volatility(data)  # Uncomment to include GJR-GARCH volatility
    }
    return volatilities

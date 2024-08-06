import numpy as np
from arch import arch_model


def calculate_std_volatility(data, window=100):
    returns = data['Log Returns'].dropna()
    return returns.rolling(window=window).std().dropna()


def calculate_ewma_volatility(data, lambda_=0.94):
    returns = data['Log Returns'].dropna()
    ewma_variance = returns ** 2

    for t in returns.index[1:]:
        ewma_variance[t] = lambda_ * ewma_variance.shift(1)[t] + (1 - lambda_) * returns[t] ** 2

    current_volatility = np.sqrt(ewma_variance)

    return current_volatility


def calculate_gjrgarch_volatility(data, scale_factor=100):
    returns = data['Log Returns'].dropna()
    returns_scaled = returns * scale_factor

    model = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, rescale=False)
    model_fit = model.fit(disp="off")
    conditional_volatility = model_fit.conditional_volatility
    conditional_volatility = conditional_volatility / scale_factor

    return conditional_volatility


def calculate_volatilities(data):
    volatilities = {
        'STD': calculate_std_volatility(data),
        'EWMA': calculate_ewma_volatility(data),
        'GJR_GARCH': calculate_gjrgarch_volatility(data)
    }
    return volatilities

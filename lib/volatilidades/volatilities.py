import numpy as np
from arch import arch_model


def std_volatility(returns, window):
    return returns.rolling(window=window).std().dropna()


def ewma_volatility(returns, lambda_=0.94):
    ewma_variance = returns ** 2

    for t in returns.index[1:]:
        ewma_variance[t] = lambda_ * ewma_variance.shift(1)[t] + (1 - lambda_) * returns[t] ** 2

    current_volatility = np.sqrt(ewma_variance)

    return current_volatility


def gjrgarch_volatility(returns):
    scale_factor = 100
    returns_scaled = returns * scale_factor

    model = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, rescale=False)
    model_fit = model.fit(disp="off")
    conditional_volatility = model_fit.conditional_volatility
    conditional_volatility = conditional_volatility / scale_factor

    return conditional_volatility


def calculate_all_volatilities(returns, window=100, lambda_=0.94):
    std_vol = std_volatility(returns, window)
    ewma_vol = ewma_volatility(returns, lambda_)
    garch_vol = gjrgarch_volatility(returns)

    # Obtener la intersección de las fechas disponibles en todas las series
    common_dates = std_vol.index.intersection(ewma_vol.index).intersection(garch_vol.index)

    # Filtrar las series para que tengan solo las fechas en común
    std_vol = std_vol.loc[common_dates]
    ewma_vol = ewma_vol.loc[common_dates]
    garch_vol = garch_vol.loc[common_dates]

    return std_vol, ewma_vol, garch_vol

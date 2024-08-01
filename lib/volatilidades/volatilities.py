import numpy as np
from arch import arch_model


def std_volatility(returns, window):
    return returns.rolling(window=window).std()


def ewma_volatility(returns, lambda_=0.94):
    returns = np.array(returns.dropna())
    ewma_variance = np.zeros_like(returns)
    ewma_variance[0] = returns[0] ** 2

    for t in range(1, len(returns)):
        ewma_variance[t] = lambda_ * ewma_variance[t - 1] + (1 - lambda_) * returns[t] ** 2

    current_volatility = np.sqrt(ewma_variance[-1])

    return current_volatility


def gjrgarch_volatility(returns):
    returns = returns.dropna()
    scale_factor = 100
    returns_scaled = returns * scale_factor

    model = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, rescale=False)
    model_fit = model.fit(disp="off")
    vol = model_fit.conditional_volatility
    last_conditional_volatility = model_fit.conditional_volatility.iloc[-1]
    last_conditional_volatility = last_conditional_volatility / scale_factor

    return last_conditional_volatility

import numpy as np


def simulate_returns(volatilidades, len_returns):
    simulated_returns = np.random.normal(loc=0, scale=volatilidades, size=len_returns)
    return simulated_returns

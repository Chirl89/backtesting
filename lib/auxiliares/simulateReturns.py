import numpy as np


def simulate_returns(volatilidades, retornos_reales):
    simulated_returns = np.random.normal(loc=0, scale=volatilidades, size=len(retornos_reales))
    return simulated_returns

import numpy as np


def var_vol(volatilidad, confidence_level):
    z_score = np.percentile(np.random.normal(0, 1, 100000), (1 - confidence_level) * 100)
    var_diario = z_score * volatilidad['VOLATILITY'].values
    return var_diario

import numpy as np


def expected_shortfall(volatilidad, nivel_confianza):
    z_score = np.percentile(np.random.normal(0, 1, 100000), (1 - nivel_confianza) * 100)
    phi_z_score = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z_score ** 2)
    es = (volatilidad['VOLATILITY'].values / (1 - nivel_confianza)) * phi_z_score
    return -es

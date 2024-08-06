import numpy as np


def es_real(data, confidence_level, start_date, end_date):

    data = data[start_date:end_date].copy()

    # Calcular el VaR al 97.5% de confianza
    var = np.percentile(data['Log Returns'].dropna(), (1 - confidence_level) * 100)

    # Calcular las excepciones (rendimientos que superan al VaR)
    if data[data['Log Returns'] < var].empty:
        exceptions = 0
    else:
        exceptions = data[data['Log Returns'] < var]
    # Calcular la ES (media de las excepciones)
    if data[data['Log Returns'] < var].empty:
        es = 0
    else:
        es = exceptions['Log Returns'].mean()
    try:
        exceptions_count = len(exceptions)
    except:
        exceptions_count = 0

    # Mostrar resultados
    return es, exceptions_count

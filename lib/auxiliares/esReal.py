import yfinance as yf
import numpy as np


def es_real(index, confidence_level, start_date, end_date):
    # Definir el periodo de tiempo

    # Descargar datos del índice SAN.MC
    data = yf.download(index, start=start_date, end=end_date, progress=False)

    # Calcular los retornos logarítmicos
    data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()

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

import sys

import pandas as pd
from volatilities import *

def calculate_rolling_volatility(returns, start_date, end_date, horizon):
    """
    Calcula la volatilidad ajustada usando EWMA para cada día en el rango de fechas especificado.

    Parámetros:
    returns (pd.Series): Serie de retornos del activo.
    start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
    horizon (int): Horizonte de riesgo en días.

    Retorna:
    pd.DataFrame: DataFrame con la fecha como índice y las volatilidades ajustadas.
    """
    # Filtrar las fechas de interés
    target_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    target_dates = target_dates[target_dates.isin(returns.index)]

    # Inicializar el diccionario para almacenar las volatilidades ajustadas
    volatilities = {
        'EWMA': [],
        'GJR_GARCH': [],
        'PERCEPTRON': [],
        'LSTM': [],
        'RANDOM_FOREST': []
    }
    n = len(target_dates)
    i = 1
    # Recorrer el rango de fechas especificado
    for target_date in target_dates:
        historical_returns = returns[:target_date - pd.Timedelta(days=horizon)]

        # Calcular la volatilidad ajustada para cada método
        volatilities['EWMA'].append(ewma_forecasting(historical_returns, horizon))
        volatilities['GJR_GARCH'].append(gjr_garch_forecasting(historical_returns, horizon))
        volatilities['PERCEPTRON'].append(perceptron_forecasting(historical_returns, horizon))
        volatilities['LSTM'].append(lstm_forecasting(historical_returns, horizon))
        volatilities['RANDOM_FOREST'].append(random_forest_forecasting(historical_returns, horizon))

        sys.stdout.write('\r')
        sys.stdout.write(f'Calculando {i} de {n} fechas - Progreso: {((i/n)*100):.2f}%')
        sys.stdout.flush()
        i += 1
    # Crear un DataFrame con la fecha como índice
    volatility_df = pd.DataFrame(volatilities, index=target_dates)

    return volatility_df

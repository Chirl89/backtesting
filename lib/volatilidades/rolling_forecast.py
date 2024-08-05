import sys

import pandas as pd
from lib.volatilidades.forecast import *


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
        'STD': [],
        'EWMA': [],
        'GJR_GARCH': [],
        'PERCEPTRON_STD': [],
        'LSTM_STD': [],
        'RANDOM_FOREST_STD': [],
        'PERCEPTRON_EWMA': [],
        'LSTM_EWMA': [],
        'RANDOM_FOREST_EWMA': [],
        'PERCEPTRON_GJR_GARCH': [],
        'LSTM_GJR_GARCH': [],
        'RANDOM_FOREST_GJR_GARCH': []
    }

    vol_std, vol_ewma, vol_gjr_garch = calculate_all_volatilities(returns=returns, window=100, lambda_=0.94)

    n = len(target_dates)
    i = 1
    # Recorrer el rango de fechas especificado
    for target_date in target_dates:
        sys.stdout.write('\r')
        sys.stdout.write(f'Calculando {i} de {n} fechas - Progreso: {(((i-1) / n) * 100):.2f}%')
        sys.stdout.flush()

        volatilities['STD'].append(vol_std[target_date])
        volatilities['EWMA'].append(vol_ewma[target_date])
        volatilities['GJR_GARCH'].append(vol_gjr_garch[target_date])

        vol_std_horizon = vol_std[:target_date - pd.Timedelta(days=horizon)]
        vol_ewma_horizon = vol_ewma[:target_date - pd.Timedelta(days=horizon)]
        vol_gjr_garch_horizon = vol_gjr_garch[:target_date - pd.Timedelta(days=horizon)]

        # Calcular la volatilidad ajustada para cada método
        volatilities['PERCEPTRON_STD'].append(perceptron_forecasting(vol_std_horizon, horizon))
        volatilities['LSTM_STD'].append(lstm_forecasting(vol_std_horizon, horizon))
        volatilities['RANDOM_FOREST_STD'].append(random_forest_forecasting(vol_std_horizon, horizon))

        volatilities['PERCEPTRON_EWMA'].append(perceptron_forecasting(vol_ewma_horizon, horizon))
        volatilities['LSTM_EWMA'].append(lstm_forecasting(vol_ewma_horizon, horizon))
        volatilities['RANDOM_FOREST_EWMA'].append(random_forest_forecasting(vol_ewma_horizon, horizon))

        volatilities['PERCEPTRON_GJR_GARCH'].append(perceptron_forecasting(vol_gjr_garch_horizon, horizon))
        volatilities['LSTM_GJR_GARCH'].append(lstm_forecasting(vol_gjr_garch_horizon, horizon))
        volatilities['RANDOM_FOREST_GJR_GARCH'].append(random_forest_forecasting(vol_gjr_garch_horizon, horizon))

        i += 1
    # Crear un DataFrame con la fecha como índice

    volatility_df = pd.DataFrame(volatilities, index=target_dates).dropna()

    sys.stdout.write('\r')
    sys.stdout.write(f'Calculadas {i-1} fechas')
    sys.stdout.flush()

    return volatility_df

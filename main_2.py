from numba import cuda
import numpy as np
from lib.data.data import Data
from lib.auxiliares.esReal import es_real
from lib.volatilidades.rolling_forecast import *


indexes = ['SAN.MC']
input_method = 'csv'
start_get_data = '2021-07-30'
end_get_data = '2024-07-30'
start_calculation_date = '2023-07-30'
end_calculation_date = '2024-07-30'
confidence_level = 0.975
horizons = [1, 10]


@cuda.jit
def compute_log_returns(adj_close, log_returns):
    i = cuda.grid(1)  # Usar un índice unidimensional
    if i < adj_close.size - 1:
        log_returns[i + 1] = np.log(adj_close[i + 1] / adj_close[i])


def process_index(index):
    input_data = Data(index, start_get_data, end_get_data, input_method)
    df = input_data.data

    # Convertir las columnas relevantes a arrays de NumPy
    adj_close = df['Adj Close'].values
    log_returns = np.zeros_like(adj_close)

    # Definir el número de threads y bloques para CUDA
    threads_per_block = 128
    blocks_per_grid = (adj_close.size + (threads_per_block - 1)) // threads_per_block

    # Ejecutar la función en la GPU
    compute_log_returns[blocks_per_grid, threads_per_block](adj_close, log_returns)

    # Volver a insertar los datos calculados en el DataFrame
    df['Log Returns'] = log_returns

    # Continuar con el resto del procesamiento
    df.dropna(inplace=True)

    es_real_result = es_real(df, confidence_level, start_calculation_date, end_calculation_date)
    volatilities = calculate_volatilities(df)

    forecast_dict = {}
    for vol, vol_data in volatilities.items():
        forecast_dict[vol] = {}
        for horizon in horizons:
            forecast_data = roll_perceptron_forecast(vol_data, start_calculation_date, end_calculation_date, horizon)
            forecast_dict[vol][horizon] = {'PERCEPTRON': forecast_data}

    return index, df, es_real_result, volatilities, forecast_dict

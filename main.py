from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.data.data import Data
from lib.auxiliares.esReal import es_real
from lib.volatilidades.rolling_forecast import *
from numba import jit
# Configuración de parámetros
indexes = ['SAN.MC']
input_method = 'csv'
start_get_data = '2021-07-30'
end_get_data = '2024-07-30'
start_calculation_date = '2023-07-30'
end_calculation_date = '2024-07-30'
confidence_level = 0.975
horizons = [1, 10]


# Función para procesar datos para un índice

def expensive_computation(vol_data, start_date, end_date, horizon):
    # Aquí iría el código que necesitas optimizar
    return roll_perceptron_forecast(vol_data, start_date, end_date, horizon)

@jit
def optimized_forecast(vol_data, start_date, end_date, horizon):
    return expensive_computation(vol_data, start_date, end_date, horizon)

def process_index(index):
    input_data = Data(index, start_get_data, end_get_data, input_method)
    df = input_data.data
    df['Log Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df.dropna(inplace=True)

    es_real_result = es_real(df, confidence_level, start_calculation_date, end_calculation_date)
    volatilities = calculate_volatilities(df)

    forecast_dict = {}
    for vol, vol_data in volatilities.items():
        forecast_dict[vol] = {}
        for horizon in horizons:
            forecast_data = optimized_forecast(vol_data, start_calculation_date, end_calculation_date, horizon)
            forecast_dict[vol][horizon] = {'PERCEPTRON': forecast_data}

    return index, df, es_real_result, volatilities, forecast_dict

# Inicialización del diccionario
index_dict = {item: {} for item in indexes}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_index, index): index for index in indexes}
    for future in as_completed(futures):
        index, df, es_real_result, volatilities, forecast_dict = future.result()
        index_dict[index]['Data'] = df
        index_dict[index]['ES Real'] = es_real_result
        index_dict[index]['Volatilities'] = volatilities
        index_dict[index]['Forecast'] = forecast_dict
        gc.collect()

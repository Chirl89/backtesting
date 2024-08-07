import sys
import gc
from copy import deepcopy
import pandas as pd
from lib.volatilidades.forecast import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Lock

# Funciones de forecasting
def roll_perceptron_forecast(vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
    forecast = {'VOLATILITY': []}
    vol_range = vol[start_date:end_date]

    for date in vol_range.index:
        vol_date = vol[:date]
        forecast_data = perceptron_forecasting(vol_date[:-horizon], horizon)
        forecast['VOLATILITY'].append(forecast_data)

        # Liberar memoria innecesaria
        del forecast_data, vol_date
        gc.collect()

        # Actualizar contador de manera segura
        """
        with lock:
            global_counter.value += 1
            progress = (global_counter.value / total_tasks) * 100
            sys.stdout.write(f'\rProgreso global: {progress:.2f}%')
            sys.stdout.flush()
        """
    forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
    del forecast, vol_range
    gc.collect()
    return forecast_df


def roll_lstm_forecast(vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
    forecast = {'VOLATILITY': []}
    vol_range = vol[start_date:end_date]

    for date in vol_range.index:
        vol_date = vol[:date]
        forecast_data = lstm_forecasting(vol_date[:-horizon], horizon)
        forecast['VOLATILITY'].append(forecast_data)

        # Liberar memoria innecesaria
        del forecast_data, vol_date
        gc.collect()

        # Actualizar contador de manera segura
        """
        with lock:
            global_counter.value += 1
            progress = (global_counter.value / total_tasks) * 100
            sys.stdout.write(f'\rProgreso global: {progress:.2f}%')
            sys.stdout.flush()
        """
    forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
    del forecast, vol_range
    gc.collect()
    return forecast_df


def roll_random_forest_forecast(vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
    forecast = {'VOLATILITY': []}
    vol_range = vol[start_date:end_date]

    for date in vol_range.index:
        vol_date = vol[:date]
        forecast_data = random_forest_forecasting(vol_date[:-horizon], horizon)
        forecast['VOLATILITY'].append(forecast_data)

        # Liberar memoria innecesaria
        del forecast_data, vol_date
        gc.collect()

        # Actualizar contador de manera segura
        """
        with lock:
            global_counter.value += 1
            progress = (global_counter.value / total_tasks) * 100
            sys.stdout.write(f'\rProgreso global: {progress:.2f}%')
            sys.stdout.flush()
        """
    forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
    del forecast, vol_range
    gc.collect()
    return forecast_df


# Función principal para ejecutar en paralelo
def run_forecast_parallel(vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(roll_perceptron_forecast, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks),
            executor.submit(roll_lstm_forecast, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks),
            executor.submit(roll_random_forest_forecast, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks)
        ]

        results = {}
        for future in as_completed(futures):
            if future == futures[0]:
                results['PERCEPTRON'] = future.result()
                with lock:
                    global_counter.value += 1
                    progress = (global_counter.value / total_tasks) * 100
                    sys.stdout.write(f'\rProgreso global forecasting: {progress:.2f}%')
                    sys.stdout.flush()
            elif future == futures[1]:
                results['LSTM'] = future.result()
                with lock:
                    global_counter.value += 1
                    progress = (global_counter.value / total_tasks) * 100
                    sys.stdout.write(f'\rProgreso global forecasting: {progress:.2f}%')
                    sys.stdout.flush()
            elif future == futures[2]:
                results['RANDOM_FOREST'] = future.result()
                with lock:
                    global_counter.value += 1
                    progress = (global_counter.value / total_tasks) * 100
                    sys.stdout.write(f'\rProgreso global forecasting: {progress:.2f}%')
                    sys.stdout.flush()

        del futures
        gc.collect()
    return results


# Función para correr el forecast sobre múltiples índices y volatilidades
def run_forecast(index_dict, forecast_dict, start_calculation_date, end_calculation_date, horizons):
    total_indices = len(index_dict)
    total_volatilities = sum(len(v['Volatilities']) for v in index_dict.values())
    total_tasks = total_indices * total_volatilities * 3  # 3 métodos de predicción (Perceptron, LSTM, Random Forest)

    with Manager() as manager:
        global_counter = manager.Value('i', 0)
        lock = manager.Lock()

        task_counter = 0

        for idx_index, (index, data) in enumerate(index_dict.items(), 1):
            forecast_dict_aux = {}
            num_volatilities = len(data['Volatilities'])

            for idx_vol, (vol, vol_data) in enumerate(data['Volatilities'].items(), 1):
                forecast_dict_aux[vol] = {}

                # print(f'\nEjecutando índice {idx_index}/{total_indices}, volatilidad {idx_vol}/{num_volatilities}')

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(run_forecast_parallel, vol_data, start_calculation_date, end_calculation_date,
                                        horizon, global_counter, lock, total_tasks): horizon for horizon in horizons}

                    for future in as_completed(futures):
                        horizon = futures[future]
                        try:
                            result = future.result()
                            forecast_dict_aux[vol][horizon] = deepcopy(result)
                            task_counter += 1
                            progress = (global_counter.value / total_tasks) * 100
                            sys.stdout.write(f'\rProgreso global forecasting: {progress:.2f}%'
                                             f'. Ejecutado con {idx_vol}/{num_volatilities} volatilidades de '
                                             f'{idx_index}/{total_indices} índices ')
                            sys.stdout.flush()

                            # Liberar memoria del resultado
                            del result
                            gc.collect()
                        except Exception as exc:
                            print(f'Error en la predicción para horizon {horizon}: {exc}')

            forecast_dict[index] = deepcopy(forecast_dict_aux)
            del forecast_dict_aux
            gc.collect()

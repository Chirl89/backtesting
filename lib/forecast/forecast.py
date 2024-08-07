import sys
import gc
from copy import deepcopy
import pandas as pd
from lib.volatilidades.forecast import perceptron_forecasting, lstm_forecasting, random_forest_forecasting
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Lock


class Forecast:
    def __init__(self, index_dict, forecast_dict, start_calculation_date, end_calculation_date, horizons):
        self.index_dict = index_dict
        self.forecast_dict = forecast_dict
        self.start_calculation_date = start_calculation_date
        self.end_calculation_date = end_calculation_date
        self.horizons = horizons

    def roll_perceptron_forecast(self, vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]

        for date in vol_range.index:
            vol_date = vol[:date]
            forecast_data = perceptron_forecasting(vol_date[:-horizon], horizon)
            forecast['VOLATILITY'].append(forecast_data)

            # Liberar memoria innecesaria
            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    def roll_lstm_forecast(self, vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]

        for date in vol_range.index:
            vol_date = vol[:date]
            forecast_data = lstm_forecasting(vol_date[:-horizon], horizon)
            forecast['VOLATILITY'].append(forecast_data)

            # Liberar memoria innecesaria
            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    @staticmethod
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

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    def run_forecast_parallel(self, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock,
                              total_tasks):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.roll_perceptron_forecast, vol, start_calculation_date, end_calculation_date,
                                horizon, global_counter, lock, total_tasks),
                executor.submit(self.roll_lstm_forecast, vol, start_calculation_date, end_calculation_date, horizon,
                                global_counter, lock, total_tasks),
                executor.submit(self.roll_random_forest_forecast, vol, start_calculation_date, end_calculation_date,
                                horizon, global_counter, lock, total_tasks)
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

    def run_forecast(self):
        total_indices = len(self.index_dict)
        total_volatilities = sum(len(v['Volatilities']) for v in self.index_dict.values())
        total_tasks = total_indices * total_volatilities * 3  # 3 métodos de predicción

        with Manager() as manager:
            global_counter = manager.Value('i', 0)
            lock = manager.Lock()

            task_counter = 0

            for idx_index, (index, data) in enumerate(self.index_dict.items(), 1):
                forecast_dict_aux = {}
                num_volatilities = len(data['Volatilities'])

                for idx_vol, (vol, vol_data) in enumerate(data['Volatilities'].items(), 1):
                    forecast_dict_aux[vol] = {}

                    with ThreadPoolExecutor() as executor:
                        futures = {
                            executor.submit(self.run_forecast_parallel, vol_data, self.start_calculation_date,
                                            self.end_calculation_date,
                                            horizon, global_counter, lock, total_tasks): horizon for horizon in
                            self.horizons}

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

                self.forecast_dict[index] = deepcopy(forecast_dict_aux)
                del forecast_dict_aux
                gc.collect()

# Uso de la clase Forecast

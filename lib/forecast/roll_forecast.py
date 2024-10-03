import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pandas as pd
from copy import deepcopy
from lib.forecast.forecast import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager
from tensorflow.keras.models import load_model
import joblib


class Forecast:
    def __init__(self, index_dict, forecast_dict, start_calculation_date, end_calculation_date, horizons):
        self.index_dict = index_dict
        self.forecast_dict = forecast_dict
        self.start_calculation_date = start_calculation_date
        self.end_calculation_date = end_calculation_date
        self.horizons = horizons

    def roll_perceptron_forecast(self, vol, start_date, end_date, horizon, index, volatility, global_counter, lock,
                                 total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]
        perceptron_model_name = f'lib/volatilidades/models/perceptron_{index}_{volatility}_{horizon}.pkl'
        scaler_path = perceptron_model_name.replace('.pkl', '_scaler.pkl')

        os.makedirs(os.path.dirname(perceptron_model_name), exist_ok=True)

        with lock:
            if not os.path.exists(perceptron_model_name):
                perceptron_train(vol[:start_date], perceptron_model_name, horizon)
            mlp_model = joblib.load(perceptron_model_name)
            scaler = joblib.load(scaler_path)

        for date in vol_range.index:
            vol_date = vol[:date]

            with lock:
                forecast_data = perceptron_forecast(vol_date[:-horizon], mlp_model, scaler, horizon)
                forecast['VOLATILITY'].append(forecast_data)

            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    def roll_lstm_forecast(self, vol, start_date, end_date, horizon, index, volatility, global_counter, lock,
                           total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]
        lstm_model_path = f'lib/volatilidades/models/lstm_{index}_{volatility}_{horizon}.keras'

        os.makedirs(os.path.dirname(lstm_model_path), exist_ok=True)

        with lock:
            if not os.path.exists(lstm_model_path):
                lstm_train(vol[:start_date], lstm_model_path, horizon)
            lstm_model = load_model(lstm_model_path)
            scaler = joblib.load(lstm_model_path.replace('.keras', '_scaler.pkl'))

        for date in vol_range.index:
            vol_date = vol[:date]

            with lock:
                forecast_data = lstm_forecast(vol_date[:-horizon], lstm_model, scaler, horizon)
                forecast['VOLATILITY'].append(forecast_data)

            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    @staticmethod
    def roll_random_forest_forecast(vol, start_date, end_date, horizon, index, volatility, global_counter, lock,
                                    total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]
        rf_model_name = f'lib/volatilidades/models/random_forest_{index}_{volatility}_{horizon}.pkl'
        scaler_path = rf_model_name.replace('.pkl', '_scaler.pkl')

        os.makedirs(os.path.dirname(rf_model_name), exist_ok=True)

        with lock:
            if not os.path.exists(rf_model_name):
                random_forest_train(vol[:start_date], rf_model_name, horizon)
            rf_model = joblib.load(rf_model_name)
            scaler = joblib.load(scaler_path)

        for date in vol_range.index:
            vol_date = vol[:date]

            with lock:
                forecast_data = random_forest_forecast(vol_date[:-horizon], rf_model, scaler, horizon)
                forecast['VOLATILITY'].append(forecast_data)

            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    def run_forecast_parallel(self, vol, start_calculation_date, end_calculation_date, horizon, index, volatility,
                              global_counter, lock, total_tasks):
        results = {}
        futures = []  # Asegurarse de que `futures` esté siempre inicializado
        try:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.roll_perceptron_forecast, vol, start_calculation_date, end_calculation_date,
                                    horizon, index, volatility, global_counter, lock, total_tasks),
                    executor.submit(self.roll_lstm_forecast, vol, start_calculation_date, end_calculation_date, horizon,
                                    index, volatility, global_counter, lock, total_tasks),
                    executor.submit(self.roll_random_forest_forecast, vol, start_calculation_date, end_calculation_date,
                                    horizon, index, volatility, global_counter, lock, total_tasks)
                ]

                for future in as_completed(futures):
                    if future == futures[0]:
                        results['PERCEPTRON'] = future.result()
                    elif future == futures[1]:
                        results['LSTM'] = future.result()
                    elif future == futures[2]:
                        results['RANDOM_FOREST'] = future.result()

                    with lock:
                        global_counter.value += 1
                    progress = (global_counter.value / total_tasks) * 100
                    sys.stdout.write(f'\rProgreso global forecasting: {progress:.2f}%')
                    sys.stdout.flush()

        except Exception as e:
            print(f"Error en la predicción: {e}")

        finally:
            del futures
            gc.collect()
        return results

    def clean_up_models(self):
        # Eliminar todos los modelos y escaladores generados
        for index in self.index_dict:
            for vol in self.index_dict[index]['Volatilities']:
                for horizon in self.horizons:
                    # Definir los nombres de los archivos de modelo
                    perceptron_model_name = f'lib/volatilidades/models/perceptron_{index}_{vol}_{horizon}.pkl'
                    lstm_model_path = f'lib/volatilidades/models/lstm_{index}_{vol}_{horizon}.keras'
                    rf_model_name = f'lib/volatilidades/models/random_forest_{index}_{vol}_{horizon}.pkl'

                    # Definir los nombres de los escaladores
                    perceptron_scaler_path = perceptron_model_name.replace('.pkl', '_scaler.pkl')
                    lstm_scaler_path = lstm_model_path.replace('.keras', '_scaler.pkl')
                    rf_scaler_path = rf_model_name.replace('.pkl', '_scaler.pkl')

                    # Borrar los archivos si existen
                    for file_path in [perceptron_model_name, perceptron_scaler_path, lstm_model_path, lstm_scaler_path,
                                      rf_model_name, rf_scaler_path]:
                        if os.path.exists(file_path):
                            os.remove(file_path)

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
                                            self.end_calculation_date, horizon, index, vol, global_counter, lock,
                                            total_tasks): horizon for horizon in
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

                                del result
                                gc.collect()
                            except Exception as exc:
                                print(f'Error en la predicción para horizon {horizon}: {exc}')

                self.forecast_dict[index] = deepcopy(forecast_dict_aux)
                del forecast_dict_aux
                gc.collect()


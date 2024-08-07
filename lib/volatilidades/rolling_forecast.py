import sys
import os
import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Lock
from copy import deepcopy

# Configuración de TensorFlow y logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').addFilter(lambda record: 'tf.function retracing' not in record.getMessage())
np.random.seed(4)


class Forecast:
    def __init__(self, index_dict, forecast_dict, start_calculation_date, end_calculation_date, horizons):
        self.index_dict = index_dict
        self.forecast_dict = forecast_dict
        self.start_calculation_date = start_calculation_date
        self.end_calculation_date = end_calculation_date
        self.horizons = horizons

    def perceptron_forecasting(self, vol, horizon, hidden_layer_sizes=(50,), random_state=42, max_iter=1000,
                               learning_rate_init=0.001, window_size=60):
        scaler = StandardScaler()
        volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))

        X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in range(len(volatilities_scaled) - window_size - horizon + 1)])
        y = volatilities_scaled[window_size + horizon - 1 : len(volatilities_scaled)].flatten()

        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                           max_iter=max_iter, learning_rate_init=learning_rate_init, verbose=0)
        mlp.fit(X, y)

        del X, y
        gc.collect()

        last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
        predicted_volatility = mlp.predict(last_window)
        del last_window
        predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

        del volatilities_scaled
        gc.collect()
        return predicted_volatility

    def lstm_forecasting(self, vol, horizon):
        set_entrenamiento = vol.to_frame()
        sc = MinMaxScaler(feature_range=(0, 1))
        set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

        time_step = 60

        X_train = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
        Y_train = set_entrenamiento_escalado[time_step + horizon - 1 : len(set_entrenamiento_escalado), 0]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        del set_entrenamiento_escalado
        gc.collect()

        modelo = Sequential()
        modelo.add(Input(shape=(X_train.shape[1], 1)))
        modelo.add(LSTM(units=10))
        modelo.add(Dense(units=1))
        modelo.add(Activation('relu'))
        modelo.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        modelo.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0, callbacks=[early_stopping])

        del X_train, Y_train
        gc.collect()

        ultimo_bloque = set_entrenamiento[-time_step:]
        ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

        prediccion_dia_horizon = modelo.predict(ultimo_bloque, verbose=0)

        del ultimo_bloque
        gc.collect()
        prediccion_dia_horizon = sc.inverse_transform(prediccion_dia_horizon)
        return prediccion_dia_horizon.flatten()[0]

    def random_forest_forecasting(self, vol, horizon, n_estimators=50, random_state=42, window_size=30):
        scaler = StandardScaler()
        volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))

        X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in range(len(volatilities_scaled) - window_size - horizon + 1)])
        y = volatilities_scaled[window_size + horizon - 1 : len(volatilities_scaled)].flatten()

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=multiprocessing.cpu_count(), verbose=0)
        rf.fit(X, y)

        del X, y
        gc.collect()

        last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
        predicted_volatility = rf.predict(last_window)
        del last_window

        predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

        del volatilities_scaled
        gc.collect()
        return predicted_volatility

    def roll_perceptron_forecast(self, vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]

        for date in vol_range.index:
            vol_date = vol[:date]
            forecast_data = self.perceptron_forecasting(vol_date[:-horizon], horizon)
            forecast['VOLATILITY'].append(forecast_data)

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
            forecast_data = self.lstm_forecasting(vol_date[:-horizon], horizon)
            forecast['VOLATILITY'].append(forecast_data)

            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    def roll_random_forest_forecast(self, vol, start_date, end_date, horizon, global_counter, lock, total_tasks):
        forecast = {'VOLATILITY': []}
        vol_range = vol[start_date:end_date]

        for date in vol_range.index:
            vol_date = vol[:date]
            forecast_data = self.random_forest_forecasting(vol_date[:-horizon], horizon)
            forecast['VOLATILITY'].append(forecast_data)

            del forecast_data, vol_date
            gc.collect()

        forecast_df = pd.DataFrame(forecast, index=vol_range.index).dropna()
        del forecast, vol_range
        gc.collect()
        return forecast_df

    def run_forecast_parallel(self, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.roll_perceptron_forecast, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks),
                executor.submit(self.roll_lstm_forecast, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks),
                executor.submit(self.roll_random_forest_forecast, vol, start_calculation_date, end_calculation_date, horizon, global_counter, lock, total_tasks)
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
                            executor.submit(self.run_forecast_parallel, vol_data, self.start_calculation_date, self.end_calculation_date,
                                            horizon, global_counter, lock, total_tasks): horizon for horizon in self.horizons}

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
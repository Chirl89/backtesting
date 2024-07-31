import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing
import logging
from lib.volatilidades.volatilities import *

logging.getLogger('tensorflow').addFilter(lambda record: 'tf.function retracing' not in record.getMessage())
np.random.seed(4)


def perceptron_forecasting(returns, horizon, hidden_layer_sizes=(100, 50), random_state=42, max_iter=5000,
                           learning_rate_init=0.0005, window_size=60, volatility_window=100):
    """
    Modelo perceptron multicapa
    :param returns:
    :param horizon:
    :param hidden_layer_sizes:
    :param random_state:
    :param max_iter:
    :param learning_rate_init:
    :param window_size:
    :param volatility_window:
    :return:
    """
    volatilities = calculate_volatility(returns, volatility_window).dropna()
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(volatilities.values.reshape(-1, 1))

    # Preparar los datos de entrenamiento
    X = []
    y = []
    for i in range(len(volatilities_scaled) - window_size - horizon + 1):
        X.append(volatilities_scaled[i:i + window_size].flatten())
        y.append(volatilities_scaled[i + window_size + horizon - 1])
    X = np.array(X)
    y = np.array(y).ravel()

    # Entrenar el modelo
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                       max_iter=max_iter, learning_rate_init=learning_rate_init, verbose=0)
    mlp.fit(X, y)

    # Predecir la volatilidad para 'horizon' días hacia adelante
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = mlp.predict(last_window)

    # Invertir la escala para obtener el valor original de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    return predicted_volatility


def lstm_forecasting(returns, horizon, window_size=60, volatility_window=100):
    # Calcular las volatilidades
    volatilities = calculate_volatility(returns, volatility_window).dropna()

    # Escalar los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(volatilities.values.reshape(-1, 1))

    # Preparar los datos de entrenamiento
    X = []
    y = []
    for i in range(len(volatilities_scaled) - window_size - horizon + 1):
        X.append(volatilities_scaled[i:i + window_size])
        y.append(volatilities_scaled[i + window_size + horizon - 1])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Redefinir las dimensiones de entrada para el LSTM
    dim_entrada = (X.shape[1], 1)  # Ventana de tiempo y 1 característica
    dim_salida = 1  # Salida unidimensional

    # Construir el modelo LSTM
    modelo = Sequential()
    modelo.add(Input(shape=dim_entrada))
    modelo.add(LSTM(units=50))
    modelo.add(Dense(units=dim_salida))
    modelo.add(Activation('relu'))
    modelo.compile(optimizer='adam', loss='mse')

    # Entrenar el modelo
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    modelo.fit(X, y, epochs=20, batch_size=32, verbose=0, callbacks=[early_stopping])

    # Predecir la volatilidad para 'horizon' días hacia adelante
    last_window = volatilities_scaled[-window_size:].reshape(1, window_size, 1)
    predicted_volatility = modelo.predict(last_window, verbose=0)

    # Invertir la escala para obtener el valor original de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility).flatten()[0]
    return predicted_volatility


def random_forest_forecasting(returns, horizon, n_estimators=100, random_state=42, window_size=30, volatility_window=100):
    volatilities = calculate_volatility(returns, volatility_window).dropna()
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(volatilities.values.reshape(-1, 1))

    # Preparar los datos de entrenamiento
    X = []
    y = []
    for i in range(len(volatilities_scaled) - window_size - horizon + 1):
        X.append(volatilities_scaled[i:i + window_size].flatten())
        y.append(volatilities_scaled[i + window_size + horizon - 1])
    X = np.array(X)
    y = np.array(y).ravel()

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=multiprocessing.cpu_count(), verbose=0)
    rf.fit(X, y)

    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = rf.predict(last_window)

    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    return predicted_volatility

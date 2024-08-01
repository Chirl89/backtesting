import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, Dropout
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
    volatilities = std_volatility(returns, volatility_window).dropna()
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


def lstm_forecasting(returns, horizon, volatility_window=100, time_step=60):
    volatilities = std_volatility(returns, volatility_window).dropna()
    set_entrenamiento = volatilities.to_frame()

    # Escalar el set de entrenamiento
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # Parámetros
    time_step = 60
      # Configurable para predicción a 'horizon' días

    # Crear los conjuntos de entrenamiento
    X_train = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in
                        range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    Y_train = np.array([set_entrenamiento_escalado[i + horizon - 1, 0] for i in
                        range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Definir dimensiones de entrada y salida
    dim_entrada = (X_train.shape[1], 1)
    dim_salida = 1  # Salida única para el día 'horizon'
    na = 20

    # Crear el modelo
    modelo = Sequential()
    modelo.add(Input(shape=dim_entrada))
    modelo.add(LSTM(units=na))
    modelo.add(Dense(units=dim_salida))
    modelo.add(Activation('relu'))
    modelo.compile(optimizer='adam', loss='mse')

    # Configurar EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    # Entrenar el modelo
    modelo.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0, callbacks=[early_stopping])

    # Realizar predicción para el día 'horizon'
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Predecir la volatilidad para el día 'horizon'
    prediccion_dia_horizon = modelo.predict(ultimo_bloque, verbose=0)

    # Invertir la escala de la predicción
    prediccion_dia_horizon = sc.inverse_transform(prediccion_dia_horizon)
    return prediccion_dia_horizon


def random_forest_forecasting(returns, horizon, n_estimators=100, random_state=42, window_size=30, volatility_window=100):
    volatilities = std_volatility(returns, volatility_window).dropna()
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

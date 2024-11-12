import os
import logging
import tensorflow as tf

# Suprimir advertencias de TensorFlow
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import gc
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import pandas as pd


def calculate_aic_bic(n_params, residuals, n_samples):
    # Log likelihood
    residual_sum_of_squares = sum(residuals ** 2)
    log_likelihood = -0.5 * n_samples * math.log(residual_sum_of_squares / n_samples)

    # AIC and BIC
    aic = 2 * n_params - 2 * log_likelihood
    bic = math.log(n_samples) * n_params - 2 * log_likelihood

    return aic, bic


# 1. Perceptron
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
import joblib
import numpy as np
import gc

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import gc


def perceptron_train(vol, model_path, horizon, index, volatility, hidden_layer_sizes=(200, 100, 50), random_state=42,
                     max_iter=5000, learning_rate_init=0.001, alpha=0.00001, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = MinMaxScaler()  # Usar MinMaxScaler
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))  # Asegurarse de usar numpy array

    # Crear secuencias de entrenamiento
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Entrenamiento del modelo
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                       random_state=random_state,
                       max_iter=max_iter,
                       learning_rate_init=learning_rate_init,
                       alpha=alpha,  # Regularización L2
                       early_stopping=True,
                       activation='tanh',  # logistic,tanh,relu
                       verbose=0)  # verbose=0 para suprimir durante el entrenamiento
    mlp.fit(X, y)

    # Número de parámetros del modelo
    n_params = sum([coef.size for coef in mlp.coefs_])

    # Cálculo de residuales
    y_pred = mlp.predict(X)
    residuals = y - y_pred

    # Calcular AIC y BIC (valores únicos)
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))

    # Guardar el modelo, el scaler, y las métricas AIC/BIC como valores únicos
    joblib.dump((mlp, aic, bic), model_path)
    joblib.dump(scaler, scaler_path)

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


def perceptron_forecast(vol, model, scaler, horizon, window_size=60):
    # Preprocesamiento de los datos
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)

    # Invertir la escala de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Asegurarse de que la predicción sea no negativa
    predicted_volatility = max(predicted_volatility, 0)

    # Liberar memoria
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility


# 2. LSTM

def lstm_train(vol, model_path, horizon, index, volatility, units=200, dropout_rate=0.3,
               epochs=50, batch_size=32, time_step=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.keras', '_scaler.pkl')

    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    scaler = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = scaler.fit_transform(set_entrenamiento)
    X = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in
                  range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    Y = set_entrenamiento_escalado[time_step + horizon - 1: len(set_entrenamiento_escalado), 0]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Entrenamiento del modelo
    model = Sequential([
        Input(shape=(time_step, 1)),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units),
        Dropout(dropout_rate),
        Dense(units=1),
        Activation('relu')
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stopping])

    # Obtener predicciones para los datos de entrenamiento
    Y_pred = model.predict(X, verbose=0)

    # Cálculo de los residuales
    residuals = Y - Y_pred.flatten()

    # Número de parámetros (pesos) del modelo
    n_params = model.count_params()

    # Calcular AIC y BIC como valores únicos
    aic, bic = calculate_aic_bic(n_params, residuals, len(Y))

    # Guardar el modelo, el scaler, y las métricas AIC/BIC
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump((aic, bic), model_path.replace('.keras', '_metrics.pkl'))

    # Liberar memoria
    del X, Y, set_entrenamiento_escalado
    gc.collect()


def lstm_forecast(vol, model, scaler, horizon, time_step=60):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    set_entrenamiento_escalado = scaler.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Limpiar sesión para evitar acumulación de gráficos
    clear_session()

    # Predicción
    prediccion_dia_horizon = model.predict(ultimo_bloque, verbose=0)
    prediccion_dia_horizon = scaler.inverse_transform(prediccion_dia_horizon)

    # Liberar memoria
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return prediccion_dia_horizon.flatten()[0]


# 3. Random Forest

def random_forest_train(vol, model_path, horizon, index, volatility, n_estimators=300, max_depth=10,
                        min_samples_split=2, min_samples_leaf=4, random_state=42, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Entrenamiento del modelo
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)

    # Cálculo de los residuales
    y_pred = rf.predict(X)
    residuals = y - y_pred

    # Número de parámetros (estimado): n_estimators * características
    n_params = n_estimators * X.shape[1]

    # Calcular AIC y BIC como valores únicos
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))

    # Guardar el modelo, el scaler, y las métricas AIC/BIC
    joblib.dump((rf, aic, bic), model_path)
    joblib.dump(scaler, scaler_path)

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


"""

"""


def random_forest_forecast(vol, model, scaler, horizon, window_size=60):
    # Preprocesamiento de los datos
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)

    # Invertir la escala de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Liberar memoria
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility

import os
import logging
import tensorflow as tf

# Suprimir advertencias de TensorFlow
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.backend import clear_session


# 1. Perceptron

def perceptron_train(vol, model_path, horizon, hidden_layer_sizes=(50, 30), random_state=42, max_iter=5000,
                     learning_rate_init=0.001, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = MinMaxScaler()  # Usar MinMaxScaler en lugar de StandardScaler
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[
               window_size + horizon - 1: len(volatilities_scaled)].flatten()  # Predecir log-volatilidades

    # Entrenamiento del modelo
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                       activation='tanh',
                       random_state=random_state,
                       max_iter=max_iter,
                       learning_rate_init=learning_rate_init,
                       alpha=0.001,  # Regularización L2
                       early_stopping=True,
                       verbose=0)
    mlp.fit(X, y)

    # Guardar el modelo y el scaler
    joblib.dump(mlp, model_path)
    joblib.dump(scaler, scaler_path)

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


def perceptron_forecast(vol, model, scaler, horizon, window_size=60):
    # Preprocesamiento de los datos
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)  # Convertir log-volatilidad a volatilidad

    # Invertir la escala de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Asegurarse de que la predicción sea no negativa
    predicted_volatility = max(predicted_volatility, 0)

    # Liberar memoria
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility


# 2. LSTM

def lstm_train(vol, model_path, horizon, time_step=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.keras', '_scaler.pkl')

    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)
    X = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in
                  range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    Y = set_entrenamiento_escalado[time_step + horizon - 1: len(set_entrenamiento_escalado), 0]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Entrenamiento del modelo
    modelo = Sequential()
    modelo.add(Input(shape=(X.shape[1], 1)))

    modelo.add(LSTM(units=50, return_sequences=True))  # Capa LSTM con 50 neuronas
    modelo.add(Dropout(0.2))  # Añadir Dropout del 20%
    modelo.add(LSTM(units=50))  # Segunda capa LSTM con 50 neuronas
    modelo.add(Dropout(0.2))  # Añadir Dropout del 20%
    modelo.add(Dense(units=1))
    modelo.add(Activation('relu'))
    modelo.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    modelo.fit(X, Y, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])

    # Guardar el modelo y el scaler
    modelo.save(model_path)
    joblib.dump(sc, scaler_path)

    # Liberar memoria
    del X, Y, set_entrenamiento_escalado
    gc.collect()


def predict(model, data):
    return model(data, training=False)


def lstm_forecast(vol, model, scaler, horizon, time_step=60):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    set_entrenamiento_escalado = scaler.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Limpiar sesión para evitar acumulación de gráficos
    clear_session()

    # Predicción
    prediccion_dia_horizon = predict(model, ultimo_bloque)
    prediccion_dia_horizon = scaler.inverse_transform(prediccion_dia_horizon)

    # Liberar memoria
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return prediccion_dia_horizon.flatten()[0]


# 3. Random Forest

def random_forest_train(vol, model_path, horizon, n_estimators=50, random_state=42, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Entrenamiento del modelo
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, verbose=0)
    rf.fit(X, y)

    # Guardar el modelo y el scaler
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


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
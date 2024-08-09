import os
import gc
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 1. Perceptron

def perceptron_train(vol, model_path, horizon, hidden_layer_sizes=(20,), random_state=42, max_iter=1000,
                     learning_rate_init=0.001, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Entrenamiento del modelo
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter,
                       learning_rate_init=learning_rate_init, verbose=0)
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
    predicted_volatility = model.predict(last_window)

    # Invertir la escala de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

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
    modelo.add(LSTM(units=5))
    modelo.add(Dense(units=1))
    modelo.add(Activation('relu'))
    modelo.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    modelo.fit(X, Y, epochs=10, batch_size=16, verbose=0, callbacks=[early_stopping])

    # Guardar el modelo y el scaler
    modelo.save(model_path)
    joblib.dump(sc, scaler_path)

    # Liberar memoria
    del X, Y, set_entrenamiento_escalado
    gc.collect()

def lstm_forecast(vol, model, scaler, horizon, time_step=60):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    set_entrenamiento_escalado = scaler.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Predicción
    prediccion_dia_horizon = model.predict(ultimo_bloque, verbose=0)
    prediccion_dia_horizon = scaler.inverse_transform(prediccion_dia_horizon)

    # Liberar memoria
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return prediccion_dia_horizon.flatten()[0]

# 3. Random Forest

def random_forest_train(vol, model_path, horizon, n_estimators=15, random_state=42, window_size=60):
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
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


def perceptron_forecasting(vol, horizon, hidden_layer_sizes=(50, ), random_state=42, max_iter=1000,
                           learning_rate_init=0.001, window_size=60):
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))

    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1 : len(volatilities_scaled)].flatten()

    # Una vez que el modelo está entrenado, puedes eliminar X e y
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                       max_iter=max_iter, learning_rate_init=learning_rate_init, verbose=0)
    mlp.fit(X, y)

    # Liberar X e y después de entrenar el modelo
    del X, y
    gc.collect()

    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = mlp.predict(last_window)
    del last_window  # Liberar last_window si no se va a utilizar más

    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Liberar volatilities_scaled si no se necesita más
    del volatilities_scaled
    gc.collect()
    return predicted_volatility


def lstm_forecasting(vol, horizon):
    set_entrenamiento = vol.to_frame()

    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    time_step = 60

    X_train = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    Y_train = set_entrenamiento_escalado[time_step + horizon - 1 : len(set_entrenamiento_escalado), 0]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Después de preparar X_train y Y_train, puedes liberar set_entrenamiento_escalado
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

    # Después de entrenar el modelo, puedes liberar X_train y Y_train
    del X_train, Y_train

    ultimo_bloque = set_entrenamiento[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    prediccion_dia_horizon = modelo.predict(ultimo_bloque, verbose=0)

    del ultimo_bloque  # Liberar ultimo_bloque después de la predicción
    gc.collect()
    prediccion_dia_horizon = sc.inverse_transform(prediccion_dia_horizon)
    return prediccion_dia_horizon.flatten()[0]


def random_forest_forecasting(vol, horizon, n_estimators=50, random_state=42, window_size=30):
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))

    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1 : len(volatilities_scaled)].flatten()

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=multiprocessing.cpu_count(), verbose=0)
    rf.fit(X, y)

    # Después de entrenar el modelo, puedes liberar X e y
    del X, y
    gc.collect()
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = rf.predict(last_window)
    del last_window  # Liberar last_window si no se va a utilizar más

    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Liberar volatilities_scaled si no se necesita más
    del volatilities_scaled
    gc.collect()
    return predicted_volatility

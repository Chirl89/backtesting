import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing
import logging


logging.getLogger('tensorflow').addFilter(lambda record: 'tf.function retracing' not in record.getMessage())
np.random.seed(4)


def ewma_forecasting(returns, horizon, lambda_=0.94):
    returns = np.array(returns.dropna())
    ewma_variance = np.zeros_like(returns)
    ewma_variance[0] = returns[0] ** 2

    for t in range(1, len(returns)):
        ewma_variance[t] = lambda_ * ewma_variance[t - 1] + (1 - lambda_) * returns[t] ** 2

    current_volatility = np.sqrt(ewma_variance[-1])
    adjusted_volatility = current_volatility * np.sqrt(horizon)

    return adjusted_volatility


def gjr_garch_forecasting(returns, horizon):
    returns = returns.dropna()
    scale_factor = 100
    returns_scaled = returns * scale_factor

    model = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, rescale=False)
    model_fit = model.fit(disp="off")

    last_conditional_volatility = np.sqrt(model_fit.conditional_volatility.iloc[-1])
    last_conditional_volatility = last_conditional_volatility / scale_factor

    adjusted_volatility = last_conditional_volatility * np.sqrt(horizon)

    return adjusted_volatility


def calculate_volatility(returns, window):
    return returns.rolling(window=window).std()


def perceptron_forecasting(returns, horizon, hidden_layer_sizes=(100, 50), random_state=42, max_iter=5000,
                           learning_rate_init=0.0005, window_size=30, volatility_window=100):
    volatilities = calculate_volatility(returns, volatility_window).dropna()
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(volatilities.values.reshape(-1, 1))

    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in range(len(volatilities_scaled) - window_size)])
    y = volatilities_scaled[window_size:].flatten()

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                       max_iter=max_iter, learning_rate_init=learning_rate_init, verbose=0)
    mlp.fit(X, y)

    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = mlp.predict(last_window)

    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]
    adjusted_volatility = predicted_volatility * np.sqrt(horizon)
    return adjusted_volatility


def lstm_forecasting(returns, horizon, volatility_window=100):
    volatilities = calculate_volatility(returns, volatility_window).dropna()
    set_entrenamiento = volatilities.to_frame()

    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    time_step = 60
    X_train = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in range(time_step, len(set_entrenamiento_escalado))])
    Y_train = np.array([set_entrenamiento_escalado[i, 0] for i in range(time_step, len(set_entrenamiento_escalado))])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    dim_entrada = (X_train.shape[1], 1)
    dim_salida = 1
    na = 50

    modelo = Sequential()
    modelo.add(Input(shape=dim_entrada))
    modelo.add(LSTM(units=na))
    modelo.add(Dense(units=dim_salida))
    modelo.add(Activation('relu'))
    modelo.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    modelo.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0, callbacks=[early_stopping])

    predicciones = []
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]

    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))
    prediccion_dia = modelo.predict(ultimo_bloque, verbose=0)
    predicciones.append(prediccion_dia[0][0])
    ultimo_bloque = np.append(ultimo_bloque[0, 1:], prediccion_dia).reshape(time_step, 1)

    predicciones = sc.inverse_transform(np.array(predicciones).reshape(-1, 1))
    adjusted_volatility = np.mean(predicciones) * np.sqrt(horizon)
    return adjusted_volatility


def random_forest_forecasting(returns, horizon, n_estimators=100, random_state=42, window_size=30, volatility_window=100):
    volatilities = calculate_volatility(returns, volatility_window).dropna()
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(volatilities.values.reshape(-1, 1))

    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in range(len(volatilities_scaled) - window_size)])
    y = volatilities_scaled[window_size:].flatten()

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=multiprocessing.cpu_count(), verbose=0)
    rf.fit(X, y)

    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = rf.predict(last_window)

    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]
    adjusted_volatility = predicted_volatility * np.sqrt(horizon)
    return adjusted_volatility

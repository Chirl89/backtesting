import os
import logging
import tensorflow as tf

# Suppress TensorFlow warnings
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
    """
    Calculate AIC and BIC for model selection.

    :param n_params: Number of parameters in the model.
    :param residuals: Residuals from model predictions.
    :param n_samples: Number of data samples.
    :return: Tuple containing AIC and BIC values.
    """
    # Calculate log likelihood based on residual sum of squares
    residual_sum_of_squares = sum(residuals ** 2)
    log_likelihood = -0.5 * n_samples * math.log(residual_sum_of_squares / n_samples)

    # Compute AIC and BIC values
    aic = 2 * n_params - 2 * log_likelihood
    bic = math.log(n_samples) * n_params - 2 * log_likelihood

    return aic, bic

def perceptron_train(vol, model_path, horizon, index, volatility, hidden_layer_sizes=(200, 100, 50), random_state=42,
                     max_iter=5000, learning_rate_init=0.001, alpha=0.00001, window_size=60):
    """
    Train an MLP (Perceptron) model for volatility forecasting.

    :param vol: Series of volatility data for training.
    :param model_path: Path to save the trained model.
    :param horizon: Forecast horizon.
    :param hidden_layer_sizes: Tuple defining the hidden layer sizes.
    :param window_size: Size of the input window for creating sequences.
    """
    # Define path for the scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Data preprocessing with MinMaxScaler
    scaler = MinMaxScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))

    # Create training sequences
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Train MLP model
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                       random_state=random_state,
                       max_iter=max_iter,
                       learning_rate_init=learning_rate_init,
                       alpha=alpha,
                       early_stopping=True,
                       activation='tanh',
                       verbose=0)
    mlp.fit(X, y)

    # Calculate model parameters and residuals
    n_params = sum([coef.size for coef in mlp.coefs_])
    y_pred = mlp.predict(X)
    residuals = y - y_pred

    # Compute AIC and BIC metrics
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))

    # Save model, scaler, and metrics
    joblib.dump((mlp, aic, bic), model_path)
    joblib.dump(scaler, scaler_path)

    # Free memory
    del X, y, volatilities_scaled
    gc.collect()

def perceptron_forecast(vol, model, scaler, horizon, window_size=60):
    """
    Make a forecast using a trained MLP (Perceptron) model.

    :param vol: Series of recent volatilities.
    :param model: Trained MLP model.
    :param scaler: Scaler used during training.
    :param horizon: Forecast horizon.
    :param window_size: Size of the input data window for prediction.
    :return: Forecasted volatility.
    """
    # Scale data and prepare input window
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)

    # Invert scaling of the prediction
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Ensure non-negative forecast
    predicted_volatility = max(predicted_volatility, 0)

    # Free memory
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility

def lstm_train(vol, model_path, horizon, index, volatility, units=200, dropout_rate=0.3,
               epochs=50, batch_size=32, time_step=60):
    """
    Train an LSTM model for volatility forecasting.

    :param vol: Series of volatility data for training.
    :param model_path: Path to save the trained model.
    :param horizon: Forecast horizon.
    :param units: Number of units in LSTM layers.
    :param dropout_rate: Dropout rate for regularization.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param time_step: Number of time steps for the input sequence.
    """
    scaler_path = model_path.replace('.keras', '_scaler.pkl')

    # Data preprocessing
    set_entrenamiento = vol.to_frame()
    scaler = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = scaler.fit_transform(set_entrenamiento)
    X = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in
                  range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    Y = set_entrenamiento_escalado[time_step + horizon - 1: len(set_entrenamiento_escalado), 0]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define and train LSTM model
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

    # Calculate residuals and compute AIC/BIC
    Y_pred = model.predict(X, verbose=0)
    residuals = Y - Y_pred.flatten()
    n_params = model.count_params()
    aic, bic = calculate_aic_bic(n_params, residuals, len(Y))

    # Save model, scaler, and metrics
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump((aic, bic), model_path.replace('.keras', '_metrics.pkl'))

    # Free memory
    del X, Y, set_entrenamiento_escalado
    gc.collect()

def lstm_forecast(vol, model, scaler, horizon, time_step=60):
    """
    Make a forecast using a trained LSTM model.

    :param vol: Series of recent volatilities.
    :param model: Trained LSTM model.
    :param scaler: Scaler used during training.
    :param horizon: Forecast horizon.
    :param time_step: Number of time steps for the input sequence.
    :return: Forecasted volatility.
    """
    # Scale data and prepare input window
    set_entrenamiento = vol.to_frame()
    set_entrenamiento_escalado = scaler.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Clear session to avoid memory accumulation
    clear_session()

    # Make prediction and invert scaling
    prediccion_dia_horizon = model.predict(ultimo_bloque, verbose=0)
    prediccion_dia_horizon = scaler.inverse_transform(prediccion_dia_horizon)

    # Free memory
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return prediccion_dia_horizon.flatten()[0]

def random_forest_train(vol, model_path, horizon, index, volatility, n_estimators=300, max_depth=10,
                        min_samples_split=2, min_samples_leaf=4, random_state=42, window_size=60):
    """
    Train a Random Forest model for volatility forecasting.

    :param vol: Series of volatility data for training.
    :param model_path: Path to save the trained model.
    :param horizon: Forecast horizon.
    :param n_estimators: Number of trees in the forest.
    :param max_depth: Maximum depth of the trees.
    :param window_size: Size of the input window for creating sequences.
    """
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Data preprocessing with StandardScaler
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)

    # Calculate residuals and compute AIC/BIC
    y_pred = rf.predict(X)
    residuals = y - y_pred
    n_params = n_estimators * X.shape[1]
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))

    # Save model, scaler, and metrics
    joblib.dump((rf, aic, bic), model_path)
    joblib.dump(scaler, scaler_path)

    # Free memory
    del X, y, volatilities_scaled
    gc.collect()

def random_forest_forecast(vol, model, scaler, horizon, window_size=60):
    """
    Make a forecast using a trained Random Forest model.

    :param vol: Series of recent volatilities.
    :param model: Trained Random Forest model.
    :param scaler: Scaler used during training.
    :param horizon: Forecast horizon.
    :param window_size: Size of the input data window for prediction.
    :return: Forecasted volatility.
    """
    # Scale data and prepare input window
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)

    # Invert scaling of the prediction
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Free memory
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility

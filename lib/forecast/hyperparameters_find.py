import os
import logging
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, KFold
import joblib
import pandas as pd
import numpy as np
from lib.forecast.forecast import calculate_aic_bic

def perceptron_hyperparameters(vol, model_path, horizon, index, volatility, param_grid=None, cv_splits=5,
                               random_state=42, max_iter=5000, window_size=60, output_dir="output"):
    """
    Perform hyperparameter tuning for an MLP model using GridSearchCV.

    :param vol: Series of volatility data for training.
    :param model_path: Path to save the trained model.
    :param horizon: Forecast horizon.
    :param param_grid: Dictionary of hyperparameters to search.
    :param cv_splits: Number of cross-validation folds.
    :param random_state: Seed for reproducibility.
    :param max_iter: Maximum number of iterations for MLP training.
    :param window_size: Size of the input data window.
    :param output_dir: Directory to save output results.
    """
    if param_grid is None:
        # Define a default grid of hyperparameters
        param_grid = {
            'hidden_layer_sizes': [(50, 30), (100, 50, 30), (150, 100, 50), (200, 100, 50)],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'alpha': [0.00001, 0.0001, 0.001, 0.01]  # L2 regularization
        }

    # Preprocess data with MinMaxScaler
    scaler = MinMaxScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Configure K-Fold and GridSearchCV
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    mlp = MLPRegressor(random_state=random_state, max_iter=max_iter, activation='tanh')

    # GridSearchCV to optimize hyperparameters
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(X, y)

    # Get the best model and hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert score to positive for interpretation

    # Save results of each parameter combination to CSV
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df[['params', 'mean_test_score']]

    output_file = os.path.join(output_dir, f"perceptron_hyperparameters_results_tanh_{volatility}_{index}_{horizon}days.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)

    # Save best model, scaler, and AIC/BIC metrics
    y_pred = best_model.predict(X)
    residuals = y - y_pred
    n_params = sum([coef.size for coef in best_model.coefs_])
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))
    joblib.dump((best_model, aic, bic), model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))

    # Write best hyperparameters and metrics to a text file
    best_params_output = os.path.join(output_dir, f"best_perceptron_hyperparameters_tanh_{volatility}_{index}_{horizon}days.txt")
    with open(best_params_output, "w") as f:
        f.write("Best Hyperparameters for Perceptron:\n")
        f.write(f"Parameters: {best_params}\n")
        f.write(f"Best MSE: {best_score:.4f}\n")

    # Free memory
    del X, y, volatilities_scaled
    gc.collect()

def lstm_hyperparameters(vol, model_path, horizon, index, volatility, param_grid=None, cv_splits=5, time_step=60,
                         output_dir="output"):
    """
    Perform manual K-Fold cross-validation and hyperparameter tuning for an LSTM model.

    :param vol: Series of volatility data for training.
    :param model_path: Path to save the trained model.
    :param horizon: Forecast horizon.
    :param param_grid: Dictionary of hyperparameters to search.
    :param cv_splits: Number of cross-validation folds.
    :param time_step: Number of time steps for training input.
    :param output_dir: Directory to save output results.
    """
    if param_grid is None:
        param_grid = {
            'units': [50, 100, 150, 200],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'epochs': [50, 100, 200]
        }

    # Preprocess data with MinMaxScaler
    scaler = MinMaxScaler()
    vol_df = vol.to_frame()
    volatilities_scaled = scaler.fit_transform(vol_df)

    # Create training sequences
    X = np.array([volatilities_scaled[i - time_step:i, 0] for i in range(time_step, len(volatilities_scaled) - horizon + 1)])
    Y = volatilities_scaled[time_step + horizon - 1: len(volatilities_scaled), 0]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    best_score = float('inf')
    best_params = None
    best_model = None
    results = []

    # Manual K-Fold cross-validation
    fold_size = len(X) // cv_splits
    for units in param_grid['units']:
        for dropout_rate in param_grid['dropout_rate']:
            for epochs in param_grid['epochs']:
                val_scores = []
                for fold in range(cv_splits):
                    # Define LSTM model
                    model = Sequential([
                        Input(shape=(time_step, 1)),
                        LSTM(units=units, return_sequences=True),
                        Dropout(dropout_rate),
                        LSTM(units=units, return_sequences=False),
                        Dropout(dropout_rate),
                        Dense(1, activation='relu')
                    ])
                    model.compile(optimizer='adam', loss='mse')

                    # Split data for training and validation
                    val_start = fold * fold_size
                    val_end = val_start + fold_size
                    X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
                    Y_train = np.concatenate([Y[:val_start], Y[val_end:]], axis=0)
                    X_val = X[val_start:val_end]
                    Y_val = Y[val_start:val_end]

                    # Train with early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_val, Y_val),
                              callbacks=[early_stopping], verbose=0)

                    # Evaluate validation loss
                    val_loss = model.evaluate(X_val, Y_val, verbose=0)
                    val_scores.append(val_loss)

                # Calculate average score and store results
                avg_score = np.mean(val_scores)
                results.append({'units': units, 'dropout_rate': dropout_rate, 'epochs': epochs, 'mse': avg_score})

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'units': units, 'dropout_rate': dropout_rate, 'epochs': epochs}
                    best_model = model

    # Save the best model, scaler, and AIC/BIC metrics
    Y_pred = best_model.predict(X, verbose=0)
    residuals = Y - Y_pred.flatten()
    n_params = best_model.count_params()
    aic, bic = calculate_aic_bic(n_params, residuals, len(Y))

    best_model.save(model_path)
    joblib.dump(scaler, model_path.replace('.keras', '_scaler.pkl'))
    joblib.dump((aic, bic), model_path.replace('.keras', '_metrics.pkl'))

    # Save results of each parameter combination to CSV
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"lstm_hyperparameters_results_{volatility}_{index}_{horizon}days.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)

    # Write best hyperparameters to a text file
    best_params_output = os.path.join(output_dir, f"best_lstm_hyperparameters_{volatility}_{index}_{horizon}days.txt")
    with open(best_params_output, "w") as f:
        f.write("Best Hyperparameters for LSTM:\n")
        f.write(f"Parameters: {best_params}\n")
        f.write(f"Best MSE: {best_score:.4f}\n")

    # Free memory
    del X, Y, volatilities_scaled
    gc.collect()

def random_forest_hyperparameters(vol, model_path, horizon, index, volatility, param_grid=None, cv_splits=5,
                                  window_size=60, random_state=42, output_dir="output"):
    """
    Perform hyperparameter tuning for a Random Forest model using GridSearchCV.

    :param vol: Series of volatility data for training.
    :param model_path: Path to save the trained model.
    :param horizon: Forecast horizon.
    :param param_grid: Dictionary of hyperparameters to search.
    :param cv_splits: Number of cross-validation folds.
    :param window_size: Size of the input data window.
    :param random_state: Seed for reproducibility.
    :param output_dir: Directory to save output results.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    # Preprocess data with StandardScaler
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Configure K-Fold and GridSearchCV
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rf = RandomForestRegressor(random_state=random_state)

    # GridSearchCV to optimize hyperparameters
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(X, y)

    # Get the best model and hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Save results of each parameter combination to CSV
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df[['params', 'mean_test_score']]

    output_file = os.path.join(output_dir, f"random_forest_hyperparameters_results_{volatility}_{index}_{horizon}days.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)

    # Save best model, scaler, and AIC/BIC metrics
    y_pred = best_model.predict(X)
    residuals = y - y_pred
    n_params = best_model.n_estimators * X.shape[1]
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))
    joblib.dump((best_model, aic, bic), model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))

    # Write the best hyperparameters to a text file
    best_params_output = os.path.join(output_dir, f"best_random_forest_hyperparameters_{volatility}_{index}_{horizon}days.txt")
    with open(best_params_output, "w") as f:
        f.write("Best Hyperparameters for Random Forest:\n")
        f.write(f"Parameters: {best_params}\n")
        f.write(f"Best MSE: {best_score:.4f}\n")

    # Free memory
    del X, y, volatilities_scaled
    gc.collect()

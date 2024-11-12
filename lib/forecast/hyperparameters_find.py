import os
import logging
import tensorflow as tf

# Suprimir advertencias de TensorFlow
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, KFold
import joblib
import pandas as pd
import numpy as np
from lib.forecast.forecast import calculate_aic_bic


def perceptron_train(vol, model_path, horizon, index, volatility, param_grid=None, cv_splits=5, random_state=42, max_iter=5000,
                     learning_rate_init=0.001, window_size=60, output_dir="output"):
    if param_grid is None:
        # Definir un grid de hiperparámetros amplio
        param_grid = {
            'hidden_layer_sizes': [(50, 30), (100, 50, 30), (150, 100, 50), (200, 100, 50)],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'alpha': [0.00001, 0.0001, 0.001, 0.01]  # Regularización L2
        }

    # Preprocesamiento de los datos
    scaler = MinMaxScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))  # Asegurarse de usar numpy array
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Configuración de K-Fold y GridSearch
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    mlp = MLPRegressor(random_state=random_state, max_iter=max_iter,
                       activation='tanh', #logistic,tanh,relu
                       )

    # GridSearchCV para optimización de hiperparámetros
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X, y)

    # Obtener el mejor modelo y el mejor conjunto de hiperparámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convertir el score negativo a positivo para interpretación

    # Guardar los resultados de cada combinación en un archivo CSV
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']  # Convertimos el score negativo en positivo para que sea interpretable
    results_df = results_df[['params', 'mean_test_score']]  # Mantener solo los parámetros y el score de test

    # Generar el nombre del archivo de resultados
    output_file = os.path.join(output_dir, f"perceptron_hyperparameters_results_tanh_{volatility}_{index}_{horizon}days.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)

    # Guardar el mejor modelo y sus métricas
    y_pred = best_model.predict(X)
    residuals = y - y_pred
    n_params = sum([coef.size for coef in best_model.coefs_])
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))
    joblib.dump((best_model, aic, bic), model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))

    # Crear un archivo TXT con el mejor conjunto de hiperparámetros y sus métricas
    best_params_output = os.path.join(output_dir, f"best_perceptron_hyperparameters_tanh_{volatility}_{index}_{horizon}days.txt")
    with open(best_params_output, "w") as f:
        f.write("Mejores Hiperparámetros para Perceptron:\n")
        f.write(f"Parámetros: {best_params}\n")
        f.write(f"Error Cuadrático Medio (MSE) de la mejor combinación: {best_score:.4f}\n")
        f.write("\nResultados de cada combinación de hiperparámetros:\n")
        for _, row in results_df.iterrows():
            f.write(f"Parámetros: {row['params']}, MSE: {row['mean_test_score']:.4f}\n")

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


def lstm_train(vol, model_path, horizon, index, volatility, param_grid=None, cv_splits=5, time_step=60, random_state=42,
               output_dir="output"):
    if param_grid is None:
        # Definimos un grid de hiperparámetros amplio
        param_grid = {
            'units': [50, 100, 150, 200],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'epochs': [50, 100, 200]
        }

    # Preprocesamiento de los datos
    scaler = MinMaxScaler()
    vol_df = vol.to_frame()  # Convertir a DataFrame
    volatilities_scaled = scaler.fit_transform(vol_df)

    # Crear secuencias de entrenamiento
    X = np.array(
        [volatilities_scaled[i - time_step:i, 0] for i in range(time_step, len(volatilities_scaled) - horizon + 1)])
    Y = volatilities_scaled[time_step + horizon - 1: len(volatilities_scaled), 0]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    best_score = float('inf')
    best_params = None
    best_model = None
    results = []

    # K-Fold Cross Validation manual
    fold_size = len(X) // cv_splits
    for units in param_grid['units']:
        for dropout_rate in param_grid['dropout_rate']:
            for epochs in param_grid['epochs']:
                val_scores = []
                for fold in range(cv_splits):
                    # Crear modelo LSTM con los hiperparámetros actuales
                    model = Sequential([
                        Input(shape=(time_step, 1)),
                        LSTM(units=units, return_sequences=True),
                        Dropout(dropout_rate),
                        LSTM(units=units, return_sequences=False),
                        Dropout(dropout_rate),
                        Dense(1, activation='relu')
                    ])
                    model.compile(optimizer='adam', loss='mse')

                    # Separar datos para entrenamiento y validación
                    val_start = fold * fold_size
                    val_end = val_start + fold_size
                    X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
                    Y_train = np.concatenate([Y[:val_start], Y[val_end:]], axis=0)
                    X_val = X[val_start:val_end]
                    Y_val = Y[val_start:val_end]

                    # Entrenamiento
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_val, Y_val),
                              callbacks=[early_stopping], verbose=0)

                    # Evaluación
                    val_loss = model.evaluate(X_val, Y_val, verbose=0)
                    val_scores.append(val_loss)

                # Calcular el puntaje promedio en todas las divisiones
                avg_score = np.mean(val_scores)
                results.append({'units': units, 'dropout_rate': dropout_rate, 'epochs': epochs, 'mse': avg_score})

                # Actualizar el mejor modelo si el puntaje actual es mejor
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'units': units, 'dropout_rate': dropout_rate, 'epochs': epochs}
                    best_model = model

    # Usar el mejor modelo para predecir
    Y_pred = best_model.predict(X, verbose=0)
    residuals = Y - Y_pred.flatten()

    # Calcular AIC y BIC
    n_params = best_model.count_params()
    aic, bic = calculate_aic_bic(n_params, residuals, len(Y))

    # Guardar el mejor modelo, el scaler, y las métricas
    best_model.save(model_path)
    joblib.dump(scaler, model_path.replace('.keras', '_scaler.pkl'))
    joblib.dump((aic, bic), model_path.replace('.keras', '_metrics.pkl'))

    # Guardar los resultados de cada combinación en un archivo CSV
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"lstm_hyperparameters_results_{volatility}_{index}_{horizon}days.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)

    # Guardar el mejor conjunto de hiperparámetros en un archivo TXT
    best_params_output = os.path.join(output_dir, f"best_lstm_hyperparameters_{volatility}_{index}_{horizon}days.txt")
    with open(best_params_output, "w") as f:
        f.write("Mejores Hiperparámetros para LSTM:\n")
        f.write(f"Parámetros: {best_params}\n")
        f.write(f"Error Cuadrático Medio (MSE) de la mejor combinación: {best_score:.4f}\n")
        f.write("\nResultados de cada combinación de hiperparámetros:\n")
        for _, row in results_df.iterrows():
            f.write(
                f"Parámetros: units={row['units']}, dropout_rate={row['dropout_rate']}, epochs={row['epochs']}, MSE: {row['mse']:.4f}\n")

    # Liberar memoria
    del X, Y, volatilities_scaled
    gc.collect()

def random_forest_train(vol, model_path, horizon, index, volatility, param_grid=None, cv_splits=5, window_size=60, random_state=42, output_dir="output"):
    if param_grid is None:
        # Definimos un grid de hiperparámetros amplio
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    # Preprocesamiento de los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Configuración de K-Fold y GridSearch
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rf = RandomForestRegressor(random_state=random_state)

    # GridSearchCV para optimización de hiperparámetros
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X, y)

    # Obtener el mejor modelo y el mejor conjunto de hiperparámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Guardar los resultados de cada combinación en un archivo CSV
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']  # Convertimos el score negativo en positivo para que sea interpretable
    results_df = results_df[['params', 'mean_test_score']]  # Mantener solo los parámetros y el score de test

    # Generar el nombre del archivo de resultados
    output_file = os.path.join(output_dir, f"random_forest_hyperparameters_results_{volatility}_{index}_{horizon}days.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)

    # Guardar el mejor modelo y sus métricas
    y_pred = best_model.predict(X)
    residuals = y - y_pred
    n_params = best_model.n_estimators * X.shape[1]
    aic, bic = calculate_aic_bic(n_params, residuals, len(y))
    joblib.dump((best_model, aic, bic), model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))

    # Crear un archivo TXT con el mejor conjunto de hiperparámetros y sus métricas
    best_params_output = os.path.join(output_dir, f"best_random_forest_hyperparameters_{volatility}_{index}_{horizon}days.txt")
    with open(best_params_output, "w") as f:
        f.write("Mejores Hiperparámetros para Random Forest:\n")
        f.write(f"Parámetros: {best_params}\n")
        f.write(f"Error Cuadrático Medio (MSE) de la mejor combinación: {best_score:.4f}\n")
        f.write("\nResultados de cada combinación de hiperparámetros:\n")
        for _, row in results_df.iterrows():
            f.write(f"Parámetros: {row['params']}, MSE: {row['mean_test_score']:.4f}\n")

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()
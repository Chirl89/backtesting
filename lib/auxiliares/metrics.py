import os
import pandas as pd
import numpy as np


# Funciones para calcular las métricas de error
def calculate_mae(real, predicted):
    return np.mean(np.abs(real - predicted))


def calculate_mse(real, predicted):
    return np.mean((real - predicted) ** 2)


def calculate_rmse(real, predicted):
    return np.sqrt(calculate_mse(real, predicted))


def calculate_mape(real, predicted):
    return np.mean(np.abs((real - predicted) / real)) * 100


def calculate_metrics(forecast_dict, index_dict, output_dir='output'):
    """
    Calcula las métricas de error (MAE, MSE, RMSE, MAPE) para cada forecast comparado con las volatilidades reales.

    :param forecast_dict: Diccionario con las volatilidades pronosticadas (forecast).
    :param index_dict: Diccionario que contiene las volatilidades reales dentro de la clave 'Volatilities'.
    :param output_dir: Directorio donde se guardará el archivo de salida Excel.
    """
    # Almacenar las métricas calculadas en una lista de diccionarios
    metrics_data = []

    # Iterar sobre el forecast_dict para obtener los valores pronosticados
    for index, volatilities in forecast_dict.items():
        for volatility, horizons in volatilities.items():
            for horizon, models in horizons.items():
                for model_name, forecast_df in models.items():
                    if not forecast_df.empty:
                        # Obtener la volatilidad real desde index_dict
                        real_vol_df = index_dict.get(index, {}).get('Volatilities', {}).get(volatility, pd.Series())

                        # Verificar que haya datos reales para comparar
                        if not real_vol_df.empty:
                            # Alinear las fechas comunes entre predicciones y valores reales
                            common_dates = forecast_df.index.intersection(real_vol_df.index)
                            real_values = real_vol_df.loc[common_dates].values
                            predicted_values = forecast_df.loc[common_dates, 'VOLATILITY'].values

                            if len(real_values) > 0 and len(predicted_values) > 0:
                                # Calcular las métricas
                                mae = calculate_mae(real_values, predicted_values)
                                mse = calculate_mse(real_values, predicted_values)
                                rmse = calculate_rmse(real_values, predicted_values)
                                mape = calculate_mape(real_values, predicted_values)

                                # Guardar las métricas en el diccionario
                                metrics_data.append({
                                    'Index': index,
                                    'Volatility': volatility,
                                    'Horizon': horizon,
                                    'Model': model_name,
                                    'MAE': mae,
                                    'MSE': mse,
                                    'RMSE': rmse,
                                    'MAPE': mape
                                })
                            else:
                                print(f"No hay valores comunes para comparar en {index}, {volatility}, {horizon}")
                        else:
                            print(f"No hay datos reales disponibles para {index}, {volatility}, {horizon}")

    # Convertir las métricas a un DataFrame
    df_metrics = pd.DataFrame(metrics_data)

    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar las métricas en un archivo Excel
    output_path = os.path.join(output_dir, 'forecast_metrics.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

    print(f"Las métricas han sido guardadas en: {output_path}")

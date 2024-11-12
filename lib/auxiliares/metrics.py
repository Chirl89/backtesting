import os
import pandas as pd
import numpy as np


# Error metric functions
def calculate_mae(real, predicted):
    """
    Calculates the Mean Absolute Error (MAE) between actual and predicted values.

    :param real: Array or Series of actual values.
    :type real: numpy.ndarray or pandas.Series
    :param predicted: Array or Series of predicted values.
    :type predicted: numpy.ndarray or pandas.Series
    :return: Mean Absolute Error (MAE) as a float.
    :rtype: float
    """
    return np.mean(np.abs(real - predicted))


def calculate_mse(real, predicted):
    """
    Calculates the Mean Squared Error (MSE) between actual and predicted values.

    :param real: Array or Series of actual values.
    :type real: numpy.ndarray or pandas.Series
    :param predicted: Array or Series of predicted values.
    :type predicted: numpy.ndarray or pandas.Series
    :return: Mean Squared Error (MSE) as a float.
    :rtype: float
    """
    return np.mean((real - predicted) ** 2)


def calculate_rmse(real, predicted):
    """
    Calculates the Root Mean Squared Error (RMSE) between actual and predicted values.

    :param real: Array or Series of actual values.
    :type real: numpy.ndarray or pandas.Series
    :param predicted: Array or Series of predicted values.
    :type predicted: numpy.ndarray or pandas.Series
    :return: Root Mean Squared Error (RMSE) as a float.
    :rtype: float
    """
    return np.sqrt(calculate_mse(real, predicted))


def calculate_mape(real, predicted):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between actual and predicted values.

    :param real: Array or Series of actual values.
    :type real: numpy.ndarray or pandas.Series
    :param predicted: Array or Series of predicted values.
    :type predicted: numpy.ndarray or pandas.Series
    :return: Mean Absolute Percentage Error (MAPE) as a float.
    :rtype: float
    """
    return np.mean(np.abs((real - predicted) / real))


def calculate_metrics(forecast_dict, index_dict, output_dir='output'):
    """
    Calculates error metrics (MAE, MSE, RMSE, MAPE) for each forecast compared to actual volatilities.

    :param forecast_dict: Dictionary with forecasted volatilities for different indices, volatilities, and horizons.
    :type forecast_dict: dict
    :param index_dict: Dictionary containing actual volatilities in the 'Volatilities' key for each index.
    :type index_dict: dict
    :param output_dir: Directory where the Excel output file with metrics will be saved.
    :type output_dir: str
    :return: None. Saves the calculated metrics in an Excel file at the specified output directory.
    """

    # Store calculated metrics in a list of dictionaries
    metrics_data = []

    # Iterate over forecast_dict to retrieve forecasted values
    for index, volatilities in forecast_dict.items():
        for volatility, horizons in volatilities.items():
            for horizon, models in horizons.items():
                for model_name, forecast_df in models.items():
                    if not forecast_df.empty:
                        # Retrieve actual volatility from index_dict
                        real_vol_df = index_dict.get(index, {}).get('Volatilities', {}).get(volatility, pd.Series())

                        # Check if there is actual data available for comparison
                        if not real_vol_df.empty:
                            # Align dates common to both forecasts and actual values
                            common_dates = forecast_df.index.intersection(real_vol_df.index)
                            real_values = real_vol_df.loc[common_dates].values
                            predicted_values = forecast_df.loc[common_dates, 'VOLATILITY'].values

                            if len(real_values) > 0 and len(predicted_values) > 0:
                                # Calculate error metrics
                                mae = calculate_mae(real_values, predicted_values)
                                mse = calculate_mse(real_values, predicted_values)
                                rmse = calculate_rmse(real_values, predicted_values)
                                mape = calculate_mape(real_values, predicted_values)

                                # Append calculated metrics to the list
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
                                print(f"No common values for comparison in {index}, {volatility}, {horizon}")
                        else:
                            print(f"No actual data available for {index}, {volatility}, {horizon}")

    # Convert metrics data to a DataFrame
    df_metrics = pd.DataFrame(metrics_data)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save metrics data to an Excel file
    output_path = os.path.join(output_dir, 'forecast_metrics.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

    print(f"Metrics have been saved to: {output_path}")

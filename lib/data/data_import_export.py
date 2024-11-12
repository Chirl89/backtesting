import os
import pandas as pd
import yfinance as yf
import joblib

class DataImporter:
    """
    Class for importing financial data, either from CSV files or using Yahoo Finance API.
    """

    def __init__(self, indexes, get_data_begin, get_data_end, method, csv_path=None):
        """
        Initialize DataImporter with parameters to define the data source and range.

        :param indexes: List of index names (tickers) or a single index.
        :param get_data_begin: Start date for data fetching.
        :param get_data_end: End date for data fetching.
        :param method: Data fetching method ('csv' for CSV files or 'yf' for Yahoo Finance).
        :param csv_path: Optional base path for CSV files (if method='csv'). Can be a string or dictionary with specific paths per index.
        """
        if isinstance(indexes, str):
            indexes = [indexes]  # Convert to list if only a single index

        self.indexes = indexes
        self.get_data_begin = get_data_begin
        self.get_data_end = get_data_end
        self.data = {}
        self.method = method
        self.csv_path = csv_path

        self.load_data()  # Load data on initialization

    def load_data(self):
        """
        Load data for each index based on the specified method (CSV or Yahoo Finance).
        """
        for index in self.indexes:
            if self.method == 'csv':
                if isinstance(self.csv_path, dict):
                    path = self.csv_path.get(index, f'input/{index}.csv')
                else:
                    path = f'input/{index}.csv'
                self.data[index] = self.get_csv_data(path)
            else:
                self.data[index] = self.get_yf_data(index)

    def get_csv_data(self, path):
        """
        Load data from a CSV file.

        :param path: Path to the CSV file.
        :return: DataFrame with data within the specified date range.
        """
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        return df[(df.index >= self.get_data_begin) & (df.index <= self.get_data_end)].dropna()

    def get_yf_data(self, index):
        """
        Fetch data from Yahoo Finance.

        :param index: Ticker symbol of the index.
        :return: DataFrame with the index data.
        """
        return yf.download(index, self.get_data_begin, self.get_data_end, progress=False)

    def return_data(self, index=None):
        """
        Return data for a specific index or all data if no index is specified.

        :param index: Index name for which to retrieve data.
        :return: DataFrame for specified index or dictionary of all data.
        """
        if index:
            return self.data.get(index, None)
        return self.data


class DataExporter:
    """
    Class for exporting index data, forecast results, and backtest results to an Excel file with multiple sheets.
    """

    def __init__(self, index_dict, forecast_dict, backtest_dict, output_dir='output'):
        """
        Initialize DataExporter with data and directory details.

        :param index_dict: Dictionary with index data, including real returns and volatilities.
        :param forecast_dict: Dictionary with forecast results.
        :param backtest_dict: Dictionary with backtest results.
        :param output_dir: Directory to save the output Excel file.
        """
        self.index_dict = index_dict
        self.forecast_dict = forecast_dict
        self.backtest_dict = backtest_dict
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, 'backtest_results.xlsx')

    def export_to_excel(self):
        """
        Structure the data and export results to an Excel file with multiple sheets.
        """
        # Lists to store structured data for each sheet
        backtest_ridge_salida_data = []
        backtest_ridge_test_data = []
        backtest_multiquantile_salida_data = []
        backtest_multiquantile_test_data = []
        backtest_fisslerziegel_salida_data = []
        backtest_fisslerziegel_test_data = []
        es_real_data = []
        volatilities_data = []
        data_data = []
        forecast_data = []
        aic_bic_data = []  # List for storing AIC/BIC metrics

        # Process backtest_dict for each index and model results
        for index, volatilities in self.backtest_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, forecast_models in horizons.items():
                    for model, results in forecast_models.items():
                        # Retrieve and separate values for Backtest Ridge outputs
                        excepciones_r, es_r = results.get('BacktestRidge - Salida', (None, None))
                        excepciones_mq, es_mq = results.get('BacktestMQ - Salida', (None, None))
                        excepciones_fz, es_fz = results.get('BacktestFZ - Salida', (None, None))

                        # Add structured data to Ridge backtest output sheet
                        backtest_ridge_salida_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'Excepciones': excepciones_r,
                            'ES': es_r
                        })

                        backtest_multiquantile_salida_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'Excepciones': excepciones_mq,
                            'ES': es_mq
                        })

                        backtest_fisslerziegel_salida_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'Excepciones': excepciones_fz,
                            'ES': es_fz
                        })

                        # Add structured data for Ridge backtest test sheet
                        backtest_ridge_test_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'BacktestRidge - Test': results.get('BacktestRidge - Test', '')
                        })

                        # Structured data for multiquantile and fisslerziegel tests
                        backtest_multiquantile_test_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'BacktestMQ - Test': results.get('BacktestMQ - Test', '')
                        })

                        backtest_fisslerziegel_test_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'BacktestRidge - Test': results.get('BacktestFZ - Test', '')
                        })

        # Capture AIC/BIC from forecast_dict
        for index, volatilities in self.forecast_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, models in horizons.items():
                    for model_name, forecast_df in models.items():
                        # Only proceed if DataFrame is not empty
                        if not forecast_df.empty:
                            # Extract unique AIC/BIC values
                            aic_value = forecast_df['AIC'].iloc[0]
                            bic_value = forecast_df['BIC'].iloc[0]
                            aic_bic_data.append({
                                'Index': index,
                                'Volatility': volatility,
                                'Horizon': horizon,
                                'Model': model_name,
                                'AIC': aic_value,
                                'BIC': bic_value
                            })

        # Capture forecast from forecast_dict
        for index, volatilities in self.forecast_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, models in horizons.items():
                    for model_name, forecast_df in models.items():
                        forecast_df_reset = forecast_df.reset_index()  # Convertir el índice en columna
                        for _, row in forecast_df_reset.iterrows():
                            row_data = row.to_dict()
                            row_data['Index'] = index
                            row_data['Volatility'] = volatility
                            row_data['Horizon'] = horizon
                            row_data['Model'] = model_name
                            forecast_data.append(row_data)
        # Process index_dict for additional data
        for index, content in self.index_dict.items():
            # Add data (e.g., prices) from index_dict
            df = content.get('Data')
            if df is not None:
                for _, row in df.iterrows():
                    row_data = row.to_dict()
                    row_data['Index'] = index
                    data_data.append(row_data)

            # Process ES Real values, handling edge cases for data structures
            es_real_values = content.get('ES Real')
            if es_real_values is not None:
                es_r, excepciones_r = es_real_values
                if not isinstance(excepciones_r, (list, pd.Series)):
                    excepciones_r = [excepciones_r]
                if not isinstance(es_r, (list, pd.Series)):
                    es_r = [es_r]
                for exc, es_val in zip(excepciones_r, es_r):
                    es_real_data.append({
                        'Index': index,
                        'Excepciones': exc,
                        'ES': es_val
                    })

            # Process volatility data in index_dict
            volatilities_dict = content.get('Volatilities')
            if volatilities_dict is not None:
                for vol_name, vol_data in volatilities_dict.items():
                    vol_data_df = vol_data.reset_index()
                    vol_data_df.rename(columns={'index': 'Date', vol_data.name: 'Volatility Value'}, inplace=True)
                    vol_data_df['Index'] = index
                    vol_data_df['Volatility'] = vol_name
                    for _, row in vol_data_df.iterrows():
                        row_data = row.to_dict()
                        volatilities_data.append(row_data)

        # Convert lists to DataFrames
        df_backtest_ridge_salida = pd.DataFrame(backtest_ridge_salida_data)
        df_backtest_ridge_test = pd.DataFrame(backtest_ridge_test_data)
        df_backtest_multiquantile_salida = pd.DataFrame(backtest_multiquantile_salida_data)
        df_backtest_multiquantile_test = pd.DataFrame(backtest_multiquantile_test_data)
        df_backtest_fisslerziegel_salida = pd.DataFrame(backtest_fisslerziegel_salida_data)
        df_backtest_fisslerziegel_test = pd.DataFrame(backtest_fisslerziegel_test_data)
        df_data = pd.DataFrame(data_data)
        df_es_real = pd.DataFrame(es_real_data)
        df_volatilities = pd.DataFrame(volatilities_data)
        df_forecast = pd.DataFrame(forecast_data)
        df_aic_bic = pd.DataFrame(aic_bic_data)

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save DataFrames to Excel with multiple sheets
        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
            df_backtest_ridge_salida.to_excel(writer, sheet_name='BacktestRidge - Salida', index=False)
            df_backtest_ridge_test.to_excel(writer, sheet_name='BacktestRidge - Test', index=False)
            df_backtest_multiquantile_salida.to_excel(writer, sheet_name='BacktestMultiquantile - Salida', index=False)
            df_backtest_multiquantile_test.to_excel(writer, sheet_name='BacktestMultiquantile - Test', index=False)
            df_backtest_fisslerziegel_salida.to_excel(writer, sheet_name='BacktestFisslerziegel - Salida', index=False)
            df_backtest_fisslerziegel_test.to_excel(writer, sheet_name='BacktestFisslerziegel - Test', index=False)
            df_data.to_excel(writer, sheet_name='Data', index=False)
            df_es_real.to_excel(writer, sheet_name='ES Real', index=False)
            df_volatilities.to_excel(writer, sheet_name='Volatilities', index=False)
            df_forecast.to_excel(writer, sheet_name='Forecast', index=False)
            df_aic_bic.to_excel(writer, sheet_name='AIC_BIC', index=False)

        print(f"El archivo se ha guardado en: {self.output_path}")

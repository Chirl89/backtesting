import os
import pandas as pd
import yfinance as yf
import joblib

class DataImporter:
    def __init__(self, indexes, get_data_begin, get_data_end, method, csv_path=None):
        """
        Constructor para la clase Data.

        :param indexes: Lista de nombres de índices (tickers) o un solo índice.
        :param get_data_begin: Fecha de inicio para obtener los datos.
        :param get_data_end: Fecha de fin para obtener los datos.
        :param method: Método para obtener los datos ('csv' o 'yf').
        :param csv_path: Ruta base opcional para archivos CSV (si method='csv'). Puede ser un string o un diccionario con rutas específicas para cada índice.
        """
        if isinstance(indexes, str):
            indexes = [indexes]  # Convertir a lista si es un solo índice

        self.indexes = indexes
        self.get_data_begin = get_data_begin
        self.get_data_end = get_data_end
        self.data = {}
        self.method = method
        self.csv_path = csv_path

        self.load_data()

    def load_data(self):
        """
        Carga los datos para cada índice según el método especificado.
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
        Obtiene datos de un archivo CSV.
        """
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        return df[(df.index >= self.get_data_begin) & (df.index <= self.get_data_end)].dropna()

    def get_yf_data(self, index):
        """
        Obtiene datos de Yahoo Finance.
        """
        return yf.download(index, self.get_data_begin, self.get_data_end, progress=False)

    def return_data(self, index=None):
        """
        Devuelve los datos de un índice específico o todos los datos si no se especifica un índice.

        :param index: Nombre del índice para el que se desea obtener los datos.
        :return: DataFrame con los datos del índice especificado o un diccionario con todos los índices.
        """
        if index:
            return self.data.get(index, None)
        return self.data


class DataExporter:
    def __init__(self, index_dict, forecast_dict, backtest_dict, output_dir='output'):
        """
        Constructor para la clase DataExporter.

        :param index_dict: Diccionario con los datos del índice, incluyendo retornos reales y volatilidades.
        :param forecast_dict: Diccionario con los resultados del forecasting.
        :param backtest_dict: Diccionario con los resultados del backtest.
        :param output_dir: Directorio donde se guardará el archivo Excel.
        """
        self.index_dict = index_dict
        self.forecast_dict = forecast_dict
        self.backtest_dict = backtest_dict
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, 'backtest_results.xlsx')

    def export_to_excel(self):
        """
        Estructura los datos y exporta los resultados a un archivo Excel con múltiples hojas.
        """
        # Crear listas para almacenar los datos estructurados para cada pestaña
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
        aic_bic_data = []  # Nueva lista para almacenar las métricas AIC/BIC

        # Iterar sobre backtest_dict y estructurar los datos
        for index, volatilities in self.backtest_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, forecast_models in horizons.items():
                    for model, results in forecast_models.items():
                        # Separar los valores de 'BacktestRidge - Salida' en dos columnas
                        excepciones_r, es_r = results.get('BacktestRidge - Salida', (None, None))
                        excepciones_mq, es_mq = results.get('BacktestMQ - Salida', (None, None))
                        excepciones_fz, es_fz = results.get('BacktestFZ - Salida', (None, None))

                        # Agregar datos a la lista para la pestaña BacktestRidge - Salida
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

                        # Agregar datos a la lista para la pestaña BacktestRidge - Test
                        backtest_ridge_test_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'BacktestRidge - Test': results.get('BacktestRidge - Test', '')
                        })

                        # Agregar datos a la lista para la pestaña BacktestRidge - Test
                        backtest_multiquantile_test_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'BacktestMQ - Test': results.get('BacktestMQ - Test', '')
                        })

                        # Agregar datos a la lista para la pestaña BacktestRidge - Test
                        backtest_fisslerziegel_test_data.append({
                            'Index': index,
                            'Volatility': volatility,
                            'Horizon': horizon,
                            'Model': model,
                            'BacktestRidge - Test': results.get('BacktestFZ - Test', '')
                        })

        # Iterar sobre forecast_dict para capturar AIC/BIC
        for index, volatilities in self.forecast_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, models in horizons.items():
                    for model_name, forecast_df in models.items():
                        # Verificamos que el DataFrame no esté vacío
                        if not forecast_df.empty:
                            # Extraemos el valor de la volatilidad (promedio o último valor) y AIC/BIC (primer valor)
                            aic_value = forecast_df['AIC'].iloc[0]  # El primer valor ya que no cambia
                            bic_value = forecast_df['BIC'].iloc[0]  # El primer valor ya que no cambia
                            aic_bic_data.append({
                                'Index': index,
                                'Volatility': volatility,
                                'Horizon': horizon,
                                'Model': model_name,
                                'AIC': aic_value,
                                'BIC': bic_value
                            })
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


        # Iterar sobre index_dict para crear los datos adicionales
        for index, content in self.index_dict.items():
            # Data
            df = content.get('Data')
            if df is not None:
                for _, row in df.iterrows():
                    row_data = row.to_dict()
                    row_data['Index'] = index
                    data_data.append(row_data)

            # ES Real (separar en dos columnas)
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

            # Volatilities (mantener el índice y los datos)
            volatilities_dict = content.get('Volatilities')
            if volatilities_dict is not None:
                for vol_name, vol_data in volatilities_dict.items():
                    vol_data_df = vol_data.reset_index()  # Convertir el índice en columna
                    vol_data_df.rename(columns={'index': 'Date', vol_data.name: 'Volatility Value'}, inplace=True)
                    vol_data_df['Index'] = index
                    vol_data_df['Volatility'] = vol_name
                    for _, row in vol_data_df.iterrows():
                        row_data = row.to_dict()
                        volatilities_data.append(row_data)

        # Convertir las listas en DataFrames
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
        df_aic_bic = pd.DataFrame(aic_bic_data)  # Nuevo DataFrame para AIC/BIC

        # Crear el directorio si no existe
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Guardar los DataFrames en un archivo Excel con múltiples hojas
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
            df_aic_bic.to_excel(writer, sheet_name='AIC_BIC', index=False)  # Nueva pestaña para AIC/BIC

        print(f"El archivo se ha guardado en: {self.output_path}")

import pandas as pd
import yfinance as yf


class Data:
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

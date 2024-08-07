import numpy as np
from lib.auxiliares.esReal import es_real
from lib.volatilidades.volatilities import *

class ProcessData:
    def __init__(self, index_instance, data_instance):
        """
        Inicializa la clase con instancias de Index y Data.

        :param index_instance: Instancia de la clase Index.
        :param data_instance: Instancia de la clase Data que contiene datos de todos los índices.
        """
        self.index_instance = index_instance
        self.data_instance = data_instance

    def fill_index_dict(self, confidence_level, start_calculation_date, end_calculation_date):
        """
        Rellena el diccionario index_dict en la instancia de Index con los datos procesados.

        :param confidence_level: Nivel de confianza para calcular el ES.
        :param start_calculation_date: Fecha de inicio para los cálculos.
        :param end_calculation_date: Fecha de fin para los cálculos.
        """
        index_dict = self.index_instance.get_index_dict()

        # Procesar cada índice
        for index in self.index_instance.indexes:
            df = self.data_instance.return_data(index)

            # Calcular Log Returns
            df['Log Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            df.dropna(inplace=True)

            # Guardar los datos procesados en index_dict
            index_dict[index]['Data'] = df
            index_dict[index]['ES Real'] = es_real(df, confidence_level, start_calculation_date, end_calculation_date)
            index_dict[index]['Volatilities'] = calculate_volatilities(df)
            index_dict[index]['Real Returns'] = df['Log Returns'][start_calculation_date:end_calculation_date]

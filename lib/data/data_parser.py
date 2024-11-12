from lib.auxiliares.esReal import es_real
from lib.volatilidades.volatilities import *


class ProcessData:
    def __init__(self, index_instance, data_instance):
        """
        Initializes the ProcessData class with instances of Index and Data.

        :param index_instance: Instance of the Index class.
        :type index_instance: Index
        :param data_instance: Instance of the Data class containing data for all indices.
        :type data_instance: Data
        """
        self.index_instance = index_instance
        self.data_instance = data_instance

    def fill_index_dict(self, confidence_level, start_calculation_date, end_calculation_date):
        """
        Populates the index_dict in the Index instance with processed data.

        :param confidence_level: Confidence level for calculating Expected Shortfall (ES).
        :type confidence_level: float
        :param start_calculation_date: Start date for calculations.
        :type start_calculation_date: str or datetime
        :param end_calculation_date: End date for calculations.
        :type end_calculation_date: str or datetime
        """
        index_dict = self.index_instance.get_index_dict()

        # Process each index
        for index in self.index_instance.indexes:
            df = self.data_instance.return_data(index)

            # Calculate Log Returns
            df['Log Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            df.dropna(inplace=True)

            # Store processed data in index_dict
            index_dict[index]['Data'] = df
            index_dict[index]['ES Real'] = es_real(df, confidence_level, start_calculation_date, end_calculation_date)
            index_dict[index]['Volatilities'] = calculate_volatilities(df)
            index_dict[index]['Real Returns'] = df['Log Returns'][start_calculation_date:end_calculation_date]

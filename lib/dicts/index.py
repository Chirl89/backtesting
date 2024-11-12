class Index:
    def __init__(self, indexes):
        """
        Constructor that takes a list of index names and generates
        the dictionaries index_dict, forecast_dict, and backtest_dict.

        :param indexes: List of index names.
        :type indexes: list of str
        """
        self.indexes = indexes
        self.index_dict = {item: {} for item in indexes}
        self.forecast_dict = {item: {} for item in indexes}
        self.backtest_dict = {item: {} for item in indexes}

    def get_index_dict(self):
        """
        Returns the index_dict containing index data.

        :return: Dictionary containing data for each index.
        :rtype: dict
        """
        return self.index_dict

    def get_forecast_dict(self):
        """
        Returns the forecast_dict containing forecast data.

        :return: Dictionary containing forecast data for each index.
        :rtype: dict
        """
        return self.forecast_dict

    def get_backtest_dict(self):
        """
        Returns the backtest_dict containing backtest results.

        :return: Dictionary containing backtest results for each index.
        :rtype: dict
        """
        return self.backtest_dict

    def __repr__(self):
        """
        Returns a string representation of the Index class, displaying
        the indexes and the contents of index_dict, forecast_dict, and backtest_dict.

        :return: String representation of the Index instance.
        :rtype: str
        """
        return (f"Index({self.indexes})\n"
                f"index_dict: {self.index_dict}\n"
                f"forecast_dict: {self.forecast_dict}\n"
                f"backtest_dict: {self.backtest_dict}")

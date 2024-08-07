class Index:
    def __init__(self, indexes):
        """
        Constructor que toma como input un vector de índices y genera
        los diccionarios index_dict, forecast_dict y backtest_dict.

        :param indexes: Lista de nombres de índices.
        """
        self.indexes = indexes
        self.index_dict = {item: {} for item in indexes}
        self.forecast_dict = {item: {} for item in indexes}
        self.backtest_dict = {item: {} for item in indexes}

    def get_index_dict(self):
        """Devuelve el diccionario index_dict."""
        return self.index_dict

    def get_forecast_dict(self):
        """Devuelve el diccionario forecast_dict."""
        return self.forecast_dict

    def get_backtest_dict(self):
        """Devuelve el diccionario backtest_dict."""
        return self.backtest_dict

    def __repr__(self):
        """Representación de la clase Index."""
        return (f"Index({self.indexes})\n"
                f"index_dict: {self.index_dict}\n"
                f"forecast_dict: {self.forecast_dict}\n"
                f"backtest_dict: {self.backtest_dict}")
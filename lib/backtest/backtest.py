import numpy as np
from lib.backtest.RidgeBacktest import EStestRidge
from lib.backtest.MQBacktest import MultiQuantileBacktest
from lib.backtest.FisslerZiegelBacktest import FisslerZiegelBacktest
from lib.auxiliares.VaR import var_vol
from lib.auxiliares.ES import expected_shortfall


class BacktestManager:
    def __init__(self, index_dict, forecast_dict, confidence_level):
        """
        Constructor para la clase BacktestManager.

        :param index_dict: Diccionario con los datos del índice, incluyendo retornos reales.
        :param forecast_dict: Diccionario con los datos de forecasting generados.
        :param confidence_level: Nivel de confianza para el backtesting (ej. 0.95).
        """
        self.index_dict = index_dict
        self.forecast_dict = forecast_dict
        self.confidence_level = confidence_level
        self.backtest_dict = {index: {} for index in index_dict}

    def run_backtest_rige(self):
        """
        Ejecuta el backtest para cada índice, volatilidad, horizonte y modelo de predicción.
        """
        for index, volatilities in self.forecast_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, forecast_models in horizons.items():
                    for model, results in forecast_models.items():
                        self._ensure_backtest_structure(index, volatility, horizon, model)
                        self._execute_backtest_ridge(index, volatility, horizon, model, results)

    def run_backtest_multiquantile(self):
        """
        Ejecuta el backtest para cada índice, volatilidad, horizonte y modelo de predicción.
        """
        for index, volatilities in self.forecast_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, forecast_models in horizons.items():
                    for model, results in forecast_models.items():
                        self._ensure_backtest_structure(index, volatility, horizon, model)
                        self._execute_backtest_multiquantile(index, volatility, horizon, model, results)

    def run_backtest_fisslerziegel(self):
        """
        Ejecuta el backtest para cada índice, volatilidad, horizonte y modelo de predicción.
        """
        for index, volatilities in self.forecast_dict.items():
            for volatility, horizons in volatilities.items():
                for horizon, forecast_models in horizons.items():
                    for model, results in forecast_models.items():
                        self._ensure_backtest_structure(index, volatility, horizon, model)
                        self._execute_backtest_fisslerziegel(index, volatility, horizon, model, results)

    def _ensure_backtest_structure(self, index, volatility, horizon, model):
        """
        Asegura que la estructura de diccionarios para el backtest esté correctamente inicializada.
        """
        if volatility not in self.backtest_dict[index]:
            self.backtest_dict[index][volatility] = {}
        if horizon not in self.backtest_dict[index][volatility]:
            self.backtest_dict[index][volatility][horizon] = {}
        if model not in self.backtest_dict[index][volatility][horizon]:
            self.backtest_dict[index][volatility][horizon][model] = {}

    def _execute_backtest_ridge(self, index, volatility, horizon, model, results):
        """
        Ejecuta el backtest para un modelo de predicción específico y guarda los resultados.
        """
        real_returns = self.index_dict[index]['Real Returns']
        predicted_volatility = results['VOLATILITY'].values

        backtest_ridge = EStestRidge(real_returns,
                                     lambda: np.random.normal(loc=0, scale=predicted_volatility,
                                                              size=len(predicted_volatility)),
                                     1 - self.confidence_level,
                                     var_vol(results, self.confidence_level),
                                     expected_shortfall(results, self.confidence_level),
                                     1000,
                                     1 - self.confidence_level)

        self.backtest_dict[index][volatility][horizon][model]['BacktestRidge'] = backtest_ridge
        self.backtest_dict[index][volatility][horizon][model]['BacktestRidge - Salida'] = backtest_ridge.backtest_out()
        self.backtest_dict[index][volatility][horizon][model][
            'BacktestRidge - Test'] = backtest_ridge.get_results_summary()

    def _execute_backtest_multiquantile(self, index, volatility, horizon, model, results):
        """
        Ejecuta el backtest para un modelo de predicción específico y guarda los resultados.
        """
        real_returns = self.index_dict[index]['Real Returns']
        predicted_volatility = results['VOLATILITY'].values

        backtest_mq = MultiQuantileBacktest(real_returns,
                                            lambda: np.random.normal(loc=0, scale=predicted_volatility,
                                                                     size=len(predicted_volatility)),
                                            1 - self.confidence_level,
                                            var_vol(results, self.confidence_level),
                                            expected_shortfall(results, self.confidence_level),
                                            1000,
                                            1 - self.confidence_level)

        self.backtest_dict[index][volatility][horizon][model]['BacktestMQ'] = backtest_mq
        self.backtest_dict[index][volatility][horizon][model]['BacktestMQ - Salida'] = backtest_mq.backtest_out()
        self.backtest_dict[index][volatility][horizon][model]['BacktestMQ - Test'] = backtest_mq.get_results_summary()

    def _execute_backtest_fisslerziegel(self, index, volatility, horizon, model, results):
        """
        Ejecuta el backtest para un modelo de predicción específico y guarda los resultados.
        """
        real_returns = self.index_dict[index]['Real Returns']
        predicted_volatility = results['VOLATILITY'].values

        backtest_fz = FisslerZiegelBacktest(real_returns,
                                            lambda: np.random.normal(
                                                loc=0, scale=predicted_volatility,
                                                size=len(predicted_volatility)),
                                            1 - self.confidence_level,
                                            var_vol(results, self.confidence_level),
                                            expected_shortfall(results, self.confidence_level),
                                            1000,
                                            1 - self.confidence_level)

        self.backtest_dict[index][volatility][horizon][model]['BacktestFZ'] = backtest_fz
        self.backtest_dict[index][volatility][horizon][model]['BacktestFZ - Salida'] = backtest_fz.backtest_out()
        self.backtest_dict[index][volatility][horizon][model]['BacktestFZ - Test'] = backtest_fz.get_results_summary()

    def get_backtest_dict(self):
        """
        Devuelve el diccionario con los resultados del backtest.
        """
        return self.backtest_dict

# Ejemplo de uso

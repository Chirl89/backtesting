import time

from lib.backtest.FisslerZiegelBacktest import FisslerZiegelBacktest
from lib.dicts.index import Index
from lib.data.data_import_export import DataImporter, DataExporter
from lib.data.data_parser import ProcessData
from lib.forecast.roll_forecast import *
from lib.backtest.backtest import BacktestManager
from lib.auxiliares.metrics import calculate_metrics

indexes = ['SAN.MC', 'BBVA.MC']# , 'SAB.MC', '^IBEX', 'BBVAE.MC', 'XTC5.MI', 'EURUSD=X']
input_method = 'yf'
start_get_data = '2021-10-04'
end_get_data = '2024-10-04'
start_calculation_date = '2024-09-04'
end_calculation_date = '2024-10-04'
confidence_level = 0.99
horizons = [1, 10]

start_time = time.time()

index_instance = Index(indexes)
data_instance = DataImporter(indexes, start_get_data, end_get_data, input_method)
process_data_instance = ProcessData(index_instance, data_instance)

process_data_instance.fill_index_dict(
    confidence_level,
    start_calculation_date,
    end_calculation_date
)

data_dict = index_instance.get_index_dict()
forecast_dict = index_instance.get_forecast_dict()

if __name__ == "__main__":
    forecast_instance = Forecast(data_dict, forecast_dict, start_calculation_date, end_calculation_date, horizons)
    forecast_instance.run_forecast()

    calculate_metrics(forecast_dict, data_dict)

    backtest_manager = BacktestManager(data_dict, forecast_dict, confidence_level)
    backtest_manager.run_backtest_multiquantile()
    backtest_manager.run_backtest_rige()
    backtest_manager.run_backtest_fisslerziegel()
    backtest_dict = backtest_manager.get_backtest_dict()
    print()
    exporter = DataExporter(data_dict, forecast_dict, backtest_dict)
    exporter.export_to_excel()

    forecast_instance.clean_up_models()

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

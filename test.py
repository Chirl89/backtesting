from lib.dicts.index import Index
from lib.data.data_import_export import DataImporter, DataExporter
from lib.data.data_parser import ProcessData
from lib.forecast.roll_forecast import *
from lib.backtest.backtest import BacktestManager


indexes = ['SAN.MC', 'BBVA.MC', 'SAB.MC', '^IBEX', 'BBVAE.MC', 'XTC5.MI', 'EURUSD=X']
input_method = 'yf'
start_get_data = '2021-09-17'
end_get_data = '2024-09-17'
start_calculation_date = '2023-09-17'
end_calculation_date = '2024-09-17'
confidence_level = 0.99
horizons = [1, 10]

index_instance = Index(indexes)
data_instance = DataImporter(indexes, start_get_data, end_get_data, input_method)
process_data_instance = ProcessData(index_instance, data_instance)

process_data_instance.fill_index_dict(
    confidence_level,
    start_calculation_date,
    end_calculation_date
)
if __name__ == "__main__":
    data_dict = index_instance.get_index_dict()
    forecast_dict = index_instance.get_forecast_dict()

    forecast_instance = Forecast(data_dict, forecast_dict, start_calculation_date, end_calculation_date, horizons)
    forecast_instance.run_single_forecast(method='random_forest')

    backtest_manager = BacktestManager(data_dict, forecast_dict, confidence_level)
    backtest_manager.run_backtest_multiquantile()
    backtest_manager.run_backtest_rige()
    backtest_manager.run_backtest_fisslerziegel()
    backtest_dict = backtest_manager.get_backtest_dict()

    exporter = DataExporter(data_dict, forecast_dict, backtest_dict)
    exporter.export_to_excel()

    forecast_instance.clean_up_models()

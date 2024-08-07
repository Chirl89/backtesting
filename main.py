from lib.dicts.index import Index
from lib.data.data import Data
from lib.data.data_parser import ProcessData

indexes = ['SAN.MC', 'BBVA.MC']
input_method = 'yf'
start_get_data = '2021-07-30'
end_get_data = '2024-07-30'
start_calculation_date = '2024-07-25'
end_calculation_date = '2024-07-30'
confidence_level = 0.975
horizons = [1, 10]

index_instance = Index(indexes)
data_instance = Data(indexes, start_get_data, end_get_data, input_method)
process_data_instance = ProcessData(index_instance, data_instance)

process_data_instance.fill_index_dict(
    confidence_level=0.95,
    start_calculation_date='2022-01-01',
    end_calculation_date='2023-12-31'
)
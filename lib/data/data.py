import pandas as pd
import yfinance as yf


class Data:
    def __init__(self, index, get_data_begin, get_data_end, method):
        self.index = index
        self.route = f'input/{index}.csv'
        self.get_data_begin = get_data_begin
        self.get_data_end = get_data_end
        self.data = []
        if method == 'csv':
            self.get_csv_data()
        else:
            self.get_yf_data()

    def get_csv_data(self):
        self.data = pd.read_csv(self.route, index_col='Date', parse_dates=True)
        self.data = self.data[(self.data.index >= self.get_data_begin) &
                              (self.data.index <= self.get_data_end)].dropna()

    def get_yf_data(self):
        self.data = yf.download(self.index, self.get_data_begin, self.get_data_end, progress=False)

    def return_data(self):
        return self.data

from datetime import datetime
from finta import TA
import pandas as pd

import abc


def fix_values(v):
    return float(v.replace('K', '000').replace('M', '000000').replace('B', '000000000')
                 .replace('.', '').replace(',', '.').replace('%', ''))


def load_data(path):
    data = pd.read_csv(path, sep=',')
    data["Date"] = pd.to_datetime(data['Date'])
    data["Open"] = data['Open'].apply(fix_values)
    data["High"] = data['High'].apply(fix_values)
    data["Low"] = data['Low'].apply(fix_values)
    data["Close"] = data['Close'].apply(fix_values)
    data["Volume"] = data['Volume'].apply(fix_values)
    data["Variation"] = data['Variation'].apply(lambda x: fix_values(x) / 100)
    data.sort_values('Date', ascending=True, inplace=True)

    data.set_index("Date", inplace=True)

    # Add indicators
    data['SMA'] = TA.SMA(data, 12)
    data['RSI'] = TA.RSI(data)
    data['OBV'] = TA.OBV(data)

    data.fillna(0, inplace=True)

    return data


class AbstractDataProvider:
    @abc.abstractmethod
    def data(self, start, end):
        pass


class CSVDataProvider(AbstractDataProvider):
    def __init__(self, path):
        self.__data = load_data(path)
        print(self.__data.index.dtype)

    def data(self, start, end):
        return self.__data.loc[start:end]

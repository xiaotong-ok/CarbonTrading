import pandas as pd
import akshare as ak
import torch
from globals import *

class Data():
    def __init__(self, symbol, period, start_date, end_date):
        self.df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date)
        self.df.rename(columns={'日期':'date', '股票代码':'code', '开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume','成交额':'amount','振幅':'amplitude','涨跌幅':'pct','涨跌额':'pct_chg','换手率':'turn'}, inplace=True)
        self.df = self.df.drop(columns='code')
        self.df = self.df.set_index('date')
        
    def get_data(self):
        return self.df

    def get_data_by_date(self, date):
        return self.df.loc[date]
    
    def save(self, df):
        return torch.save(df, DATA_PATH)

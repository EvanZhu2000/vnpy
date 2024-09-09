from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
today_date = datetime.today()


if __name__ == "__main__":    
    set1 = {'IF','IH','IC','IM','T','TS','TF','TL'}
    set2 = {'AG','AU','SC','EC','CU'}
    set3 = set({}) # self definition
    to_drop_list = list(set1|set2|set3)
    df = mysqlservice.select('trading_schedule', 'order by date desc', strategy='dom').iloc[0]
    potential_trading_series = pd.Series(df['symbol'].split(',')).str.split('.').str[0]
    trading_list = potential_trading_series.loc[~potential_trading_series.str[:-4].isin(to_drop_list)].values
    mysqlservice.insert('trading_schedule',ignore=True,date = df['date'], symbol = ','.join(trading_list), strategy='strategy2', sc_symbol='dom')
    mysqlservice.close()
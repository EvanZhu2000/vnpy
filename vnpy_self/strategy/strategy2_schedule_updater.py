from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
from datetime import datetime
import pandas as pd

today_date = datetime.today()


if __name__ == "__main__":    
    set1 = {'IF','IH','IC','IM','T','TS','TF','TL'}
    set2 = {'AG','AU','SC','EC','CU'}
    set3 = {}  # self definition
    to_drop_list = list(set1|set2|set3)
    df = mysqlservice.select('trading_schedule', 'order by date desc', strateg='dom').iloc[0]
    potential_trading_list = (pd.Series(df['symbol'].split(',')).str[:-4]).tolist()
    trading_list = list(set(potential_trading_list) - set(to_drop_list))
    mysqlservice.insert('trading_schedule',date = df['date'], symbol = ','.join(trading_list), strategy='strategy2', sc_symbol='strategy2')
    mysqlservice.close()
from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
mysqlservice.init_connection()
import pandas as pd
import sys

def run(today_date:str): 
    set1 = {'IF','IH','IC','IM','T','TS','TF','TL'}
    set2 = {'AG','AU','SC','EC','CU'}
    set3 = {'BC', 'BR', 'CJ', 'EC', 'IH', 'NR', 'PB', 'SC', 'SH', 'SN', 'TL', 'TS'} # self definition
    to_drop_list = list(set1|set2|set3)
    df = mysqlservice.select('trading_schedule', today = today_date, strategy='dom').iloc[0]
    potential_trading_series = pd.Series(df['symbol'].split(',')).str.split('.').str[0]
    trading_list = potential_trading_series.loc[~potential_trading_series.str[:-4].isin(to_drop_list)].values
    mysqlservice.insert('trading_schedule',ignore=True,
                        today = today_date,date = df['date'], 
                        symbol = ','.join(trading_list), strategy='strategy2', sc_symbol='dom')
    mysqlservice.close()

if __name__ == "__main__":
    # The input today_date needs to be the real date at next settlement date start, in the format of YYYY-MM-DD
    today_date = sys.argv[1]
    run(today_date)




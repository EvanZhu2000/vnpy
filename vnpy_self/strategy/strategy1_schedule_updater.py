import pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import re
from vnpy_self.data_and_db.db_setting import db_setting
import mysql.connector
mydb = mysql.connector.connect(
  host= db_setting['host'],
  user= db_setting['user'],
  password= db_setting['password']
)
mycursor = mydb.cursor()

def get_from_akshare():
    futures_contract_info_cffex_df = None
    for i in range(0,15):
        try_this_date = (datetime.today() - timedelta(days = i)).strftime('%Y%m%d')
        try:
            futures_contract_info_cffex_df = ak.futures_contract_info_cffex(date=try_this_date)
            break
        except:
            continue
    return futures_contract_info_cffex_df
    


def update_strategy1_schedule(sc_symbol, rollover_days=7) -> None:
    '''
    return whether strategy1 needs rollover, available to run before trading start 
    '''
    futures_contract_info_cffex_df = get_from_akshare()
    if futures_contract_info_cffex_df is None:
        raise Exception('Cannot query future data')
    else:
        print(f"Query this date - {futures_contract_info_cffex_df['查询交易日'].loc[0].strftime('%Y%m%d')}")
        
    # Gets quarterly contracts info
    suitable_df = futures_contract_info_cffex_df.loc[(futures_contract_info_cffex_df['合约代码'].str.startswith(str(" ".join(re.findall("[a-zA-Z]+", sc_symbol))))) & (futures_contract_info_cffex_df['合约月份'].astype(int) % 3 == 0)].sort_values(by=['最后交易日'])
    should_trade_series = suitable_df.loc[pd.to_datetime(suitable_df['最后交易日'])>datetime.today() + timedelta(days = rollover_days)].iloc[:2]['合约代码'].reset_index().drop(['index'],axis=1).squeeze()
    futures_fees_info_df = ak.futures_fees_info()
    should_trade_series = futures_fees_info_df.loc[futures_fees_info_df["合约"].isin(should_trade_series)][["合约","交易所"]].agg('.'.join, axis=1).reset_index().drop('index',axis=1).squeeze()

    trading_df = pd.read_sql_query(f"SELECT * FROM vnpy.trading_schedule where strategy = 'strategy1' and sc_symbol='{sc_symbol}' order by date desc", mydb)
    if trading_df.shape[0] == 0:  # there isn't any available contracts yet
        mycursor.execute(f"INSERT INTO `vnpy`.`trading_schedule` (`date`, `symbol`, `strategy`, `sc_symbol`) VALUES ('{datetime.strftime(datetime.today(),'%Y-%m-%d')}', '{','.join(should_trade_series.tolist())}', 'strategy1', '{sc_symbol}');")
        mydb.commit()
        return
    trading_series = pd.Series(trading_df.loc[0]['symbol'].split(','), name='合约代码')  # the last trading symbol
        
    rollover_flag = False
    if not trading_series.equals(should_trade_series):
        if should_trade_series.shape[0] == 2:
            if (should_trade_series.str.extract('(\d+)') <= trading_series.str.extract('(\d+)')).sum()[0] == 0:
                rollover_flag = True
            else:
                raise Exception('The rollover target is earlier than the current contract')
        else:
            raise Exception('Do not have should_trade_series data???')
    
    if rollover_flag:
        mycursor.execute(f"INSERT INTO `vnpy`.`trading_schedule` (`date`, `symbol`, `strategy`, `sc_symbol`) VALUES ('{datetime.strftime(datetime.today(),'%Y-%m-%d')}', '{','.join(should_trade_series.tolist())}', 'strategy1', '{sc_symbol}');")
        mydb.commit()
            
            
if __name__ == "__main__":
    update_strategy1_schedule('IH_2')
    mycursor.close()
    mydb.close()
    

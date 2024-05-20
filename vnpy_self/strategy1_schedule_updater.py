import pandas as pd
from datetime import datetime, timedelta
import akshare as ak

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
    


def update_strategy1_schedule(strategy_symbol, rollover_days=5) -> None:
    '''
    return whether strategy1 needs rollover, available to run before trading start 
    only focus on IH for now
    '''
    futures_contract_info_cffex_df = get_from_akshare()
    if futures_contract_info_cffex_df is None:
        raise Exception('Cannot query future data')
    else:
        print(f"Query this date - {futures_contract_info_cffex_df['查询交易日'].loc[0].strftime('%Y%m%d')}")
        
    # Gets quarterly contracts info
    suitable_df = futures_contract_info_cffex_df.loc[(futures_contract_info_cffex_df['合约代码'].str.startswith(strategy_symbol)) & (futures_contract_info_cffex_df['合约月份'].astype(int) % 3 == 0)].sort_values(by=['最后交易日'])
    should_trade_series = suitable_df.loc[pd.to_datetime(suitable_df['最后交易日'])>datetime.today() + timedelta(days = rollover_days)].iloc[:2]['合约代码'].reset_index().drop(['index'],axis=1).squeeze()
    trading_instrument_dict = pd.read_excel(r'C:\\veighna_studio\\Lib\\site-packages\\vnpy_self\\strategy1_schedule.xlsx', sheet_name=None)
    trading_series = pd.Series(trading_instrument_dict['IH']['instrument'].values[-1].split(','), name='合约代码')  # the last trading instrument
        
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
        trading_instrument_dict['IH'].loc[trading_instrument_dict['IH'].shape[0]] = [datetime.strftime(datetime.today(),'%Y-%m-%d') , ','.join(should_trade_series.tolist())]
        with pd.ExcelWriter(r'C:\\veighna_studio\\Lib\\site-packages\\vnpy_self\\strategy1_schedule.xlsx', mode="a", engine="openpyxl", if_sheet_exists='replace') as writer:
            trading_instrument_dict['IH'].set_index('date').to_excel(writer, sheet_name="IH")  
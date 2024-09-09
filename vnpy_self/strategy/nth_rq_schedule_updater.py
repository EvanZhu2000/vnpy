import numpy as np
from datetime import datetime, timedelta
from vnpy_self.data_and_db.db_setting import db_setting

import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()

def symbol_rq2vnpy(l):
    return all_data.loc[all_data['order_book_id'].isin(l)][['trading_code','exchange']].apply(lambda x: '.'.join(x), axis = 1).values

if __name__ == "__main__":
    all_data = all_instruments(type='Future', market='cn', date=None)
    all_futures_df = all_data.loc[(all_data['order_book_id'].str.contains('888'))&(all_data['maturity_date'] == '0000-00-00')]  
    # This is how rq calculate the get_dominant() contracts
    symb_list = (all_futures_df['order_book_id']).str[:-3].tolist()
    # stock index futures and some very illiquid products have wrong rules (RQ mistake)
    # symb_list = list(set(symb_list) - set(['IC', 'IF', 'IH', 'IM', 'LR', 'RI', 'RS', 'WH']))
    symb_list = list(set(symb_list) - set(['IC', 'IF', 'IH', 'IM']))

    today_str = datetime.today().strftime('%Y-%m-%d')
    rq_next_trading_day_dom_list,rq_next_trading_day_dom2_list = [],[]
    next_trading_day = get_next_trading_date(today_str).strftime('%Y-%m-%d')
    for i in all_futures_df['underlying_symbol'].values:
        dom_contract = futures.get_dominant(i,today_str,rule=0,rank=1).values[0]
        df = get_price((futures.get_contracts(i,today_str)),today_str,today_str,'1d').sort_index().droplevel(1)
        candidates = df.iloc[np.where(df.index.get_level_values(0) == dom_contract)[0][0]+1:]['open_interest']>1.1*df.iloc[np.where(df.index.get_level_values(0) == dom_contract)[0][0]]['open_interest']
        if len(candidates[candidates]) != 0:
            next_trading_day_dom_contract = candidates[candidates].index[0]
        else:
            next_trading_day_dom_contract = dom_contract
        rq_next_trading_day_dom_list.append(next_trading_day_dom_contract)
        
        if i in symb_list:
            dom2_contract = futures.get_dominant(i,today_str,rule=0,rank=2).values[0]
            can1 = df.iloc[np.where(df.index.get_level_values(0) == next_trading_day_dom_contract)[0][0]+1:]['open_interest']
            can2 = df.iloc[np.where(df.index.get_level_values(0) == dom2_contract)[0][0]:]['open_interest']
            can = can1.loc[can1.index.intersection(can2.index)]
            rq_next_trading_day_dom2_list.append(can.idxmax())

    next_trading_day_dom_list = symbol_rq2vnpy(rq_next_trading_day_dom_list)
    next_trading_day_dom2_list = symbol_rq2vnpy(rq_next_trading_day_dom2_list)
    mysqlservice.insert('trading_schedule',date = next_trading_day, symbol = ','.join(next_trading_day_dom_list), strategy = 'dom', sc_symbol = 'rq_dom')
    mysqlservice.insert('trading_schedule',date = next_trading_day, symbol = ','.join(next_trading_day_dom2_list), strategy = 'dom2', sc_symbol = 'rq_dom2')
    
    for i in range(len(next_trading_day_dom_list)):
        symb = next_trading_day_dom_list[i]
        rq_symb = rq_next_trading_day_dom_list[i]
        mysqlservice.insert('trading_hours', date = next_trading_day, 
                            rqsymbol = rq_symb, symbol = symb, 
                            trading_hours = get_trading_hours(rq_symb, next_trading_day), timezone = 'Asia/Shanghai')
        
    for i in range(len(next_trading_day_dom2_list)):
        symb = next_trading_day_dom2_list[i]
        rq_symb = rq_next_trading_day_dom2_list[i]
        mysqlservice.insert('trading_hours', date = next_trading_day, 
                            rqsymbol = rq_symb, symbol = symb, 
                            trading_hours = get_trading_hours(rq_symb, next_trading_day), timezone = 'Asia/Shanghai')
    mysqlservice.close()
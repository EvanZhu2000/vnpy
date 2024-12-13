import sys

import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
mysqlservice.init_connection()

def symbol_rq2vnpy(l, all_data):
    return all_data.loc[all_data['order_book_id'].isin(l)][['trading_code','exchange']].apply(lambda x: '.'.join(x), axis = 1).values

def run(today_str:str): 
    all_data = all_instruments(type='Future', market='cn', date=None)
    all_futures_df = all_data.loc[(all_data['order_book_id'].str.contains('888'))&(all_data['maturity_date'] == '0000-00-00')]  
    # This is how rq calculate the get_dominant() contracts
    symb_list = (all_futures_df['order_book_id']).str[:-3].tolist()
    # stock index futures and some very illiquid products have wrong rules (RQ mistake)
    # symb_list = list(set(symb_list) - set(['IC', 'IF', 'IH', 'IM', 'LR', 'RI', 'RS', 'WH']))
    symb_list = list(set(symb_list) - set(['IC', 'IF', 'IH', 'IM']))

    next_trading_day = get_next_trading_date(today_str).strftime('%Y-%m-%d')
    rq_today_dom_list,rq_today_dom2_list = [],[]
    rq_next_trading_day_dom_list,rq_next_trading_day_dom2_list = [],[]
    
    for i in all_futures_df['underlying_symbol'].values:
        dom_contract_next_day = futures.get_dominant(i,next_trading_day,rule=0,rank=1).values[0]
        rq_next_trading_day_dom_list.append(dom_contract_next_day)
        dom_contract_today = futures.get_dominant(i,today_str,rule=0,rank=1).values[0]
        rq_today_dom_list.append(dom_contract_today)
        
        if i in symb_list:
            dom2_contract_next_day = futures.get_dominant(i,next_trading_day,rule=0,rank=2).values[0]
            rq_next_trading_day_dom2_list.append(dom2_contract_next_day)
            dom2_contract_today = futures.get_dominant(i,today_str,rule=0,rank=2).values[0]
            rq_today_dom2_list.append(dom2_contract_today)

    next_trading_day_dom_list = symbol_rq2vnpy(rq_next_trading_day_dom_list, all_data)
    next_trading_day_dom2_list = symbol_rq2vnpy(rq_next_trading_day_dom2_list, all_data)
    today_dom_list =  symbol_rq2vnpy(rq_today_dom_list, all_data)
    today_dom2_list = symbol_rq2vnpy(rq_today_dom2_list, all_data)
    mysqlservice.insert('trading_schedule', ignore=True, today = today_str, date = next_trading_day, symbol = ','.join(rq_next_trading_day_dom_list), strategy = 'dom', sc_symbol = 'rq_dom')
    mysqlservice.insert('trading_schedule', ignore=True, today = today_str, date = next_trading_day, symbol = ','.join(rq_next_trading_day_dom2_list), strategy = 'dom2', sc_symbol = 'rq_dom2')
    
    for i in range(len(next_trading_day_dom_list)):
        symb = next_trading_day_dom_list[i]
        rq_symb = rq_next_trading_day_dom_list[i]
        mysqlservice.insert('trading_hours', ignore=True, date = next_trading_day, 
                            rqsymbol = rq_symb, symbol = symb, 
                            trading_hours = get_trading_hours(rq_symb, next_trading_day), timezone = 'Asia/Shanghai')
        
    for i in range(len(next_trading_day_dom2_list)):
        symb = next_trading_day_dom2_list[i]
        rq_symb = rq_next_trading_day_dom2_list[i]
        mysqlservice.insert('trading_hours', ignore=True, date = next_trading_day, 
                            rqsymbol = rq_symb, symbol = symb, 
                            trading_hours = get_trading_hours(rq_symb, next_trading_day), timezone = 'Asia/Shanghai')
        
    for i in range(len(today_dom_list)):
        symb = today_dom_list[i]
        rq_symb = rq_today_dom_list[i]
        mysqlservice.insert('trading_hours', ignore=True, date = next_trading_day, 
                            rqsymbol = rq_symb, symbol = symb, 
                            trading_hours = get_trading_hours(rq_symb, next_trading_day), timezone = 'Asia/Shanghai')
        
    for i in range(len(today_dom2_list)):
        symb = today_dom2_list[i]
        rq_symb = rq_today_dom2_list[i]
        mysqlservice.insert('trading_hours', ignore=True, date = next_trading_day, 
                            rqsymbol = rq_symb, symbol = symb, 
                            trading_hours = get_trading_hours(rq_symb, next_trading_day), timezone = 'Asia/Shanghai')
    mysqlservice.close()
    
if __name__ == "__main__":
    # The input today_date needs to be the real date at next settlement date start, in the format of YYYY-MM-DD
    today_date = sys.argv[1]
    run(today_date)

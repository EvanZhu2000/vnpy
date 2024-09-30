## NTH means only can run this script in non-trading hours
import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')     


from vnpy_portfoliostrategy.mysqlservice import MysqlService
import sys
import pandas as pd
from tqdm import tqdm


def run_price(start_date_str, end_date_str, record_freq = '1m'):
    sqlservice = MysqlService()
    data = all_instruments(type='Future', market='cn', date=end_date_str)
    instrument_list = data['order_book_id'].loc[(data['order_book_id'].str.contains('99') == False) &
                    (data['order_book_id'].str.contains('88') == False)].dropna().unique().tolist()
    info = get_price(instrument_list, 
                    start_date = start_date_str, 
                    end_date   = end_date_str, 
                    frequency=record_freq)
    aaa = info.reset_index().set_index('order_book_id').join(data[['order_book_id', 'exchange','trading_code']].set_index('order_book_id')).reset_index().drop(['order_book_id'],axis=1).rename(columns={"total_turnover": "turnover", "open":"open_price","low":"low_price","high":"high_price","close":"close_price","trading_code":"symbol"})
    aaa['interval'] = record_freq
    aaa = aaa[['symbol', 'exchange', 'datetime', 'interval', 'volume', 'turnover', 'open_interest', 'open_price', 'high_price', 'low_price', 'close_price']]
    sqlservice.insert_rq(aaa, 'dbbardata', ignore=True)
    sqlservice.close()
    
def run_member_rank(start_date_str, end_date_str):
    sqlservice = MysqlService()
    
    data = all_instruments(type='Future', market='cn', date=end_date_str)
    instrument_list = data['order_book_id'].loc[(data['order_book_id'].str.contains('99') == False) &
                    (data['order_book_id'].str.contains('88') == False)].dropna().unique().tolist()
    
    res = pd.DataFrame()
    for ind in tqdm(instrument_list):
        long = futures.get_member_rank(ind,
                                start_date=start_date_str,
                                end_date=end_date_str,
                                rank_by='long')
        short = futures.get_member_rank(ind,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            rank_by='short')
        volume = futures.get_member_rank(ind,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        rank_by='volume')
        if long is not None:
            long['rank_by']='long'
            res = pd.concat([res,long])
        if short is not None:
            short['rank_by']='short'
            res = pd.concat([res,short])
        if volume is not None:
            volume['rank_by']='volume'
            res = pd.concat([res,volume])
    res = res.reset_index()
    sqlservice.insert_rq(res, 'dbmemberrank', ignore=True)
    sqlservice.close()
    
def run_dominant(start_date_str, end_date_str):
    sqlservice = MysqlService()
    
    data = all_instruments(type='Future', market='cn', date=end_date_str)
    instrument_parent_list = data['underlying_symbol'].unique()
    
    res = pd.DataFrame()
    for ind in tqdm(instrument_parent_list):
        rule0rank1 = futures.get_dominant(ind, start_date_str,end_date_str,rule=0,rank=1)
        rule0rank2 = futures.get_dominant(ind, start_date_str,end_date_str,rule=0,rank=2)
        
        if rule0rank1 is not None:
            rule0rank1 = rule0rank1.to_frame()
            rule0rank1['rulerank'] = 'rule0rank1'
            res = pd.concat([res, rule0rank1])
            
        if rule0rank2 is not None:
            rule0rank2 = rule0rank2.to_frame()
            rule0rank2['rulerank'] = 'rule0rank2'
            res = pd.concat([res, rule0rank2])
    res = res.reset_index()
    sqlservice.insert_rq(res, 'dbdominant', ignore=True)
    sqlservice.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Wrong number of input!!")
    strOfFields = sys.argv[1]
    listOfFields= [x for x in strOfFields.split(',')]
    start_date_str = sys.argv[2]
    end_date_str = sys.argv[3]
    for l in listOfFields:
        if l == 'price':
            run_price(start_date_str, end_date_str)
        if l == 'member_rank':
            run_member_rank(start_date_str, end_date_str)
        if l == 'dominant':
            run_dominant(start_date_str, end_date_str)
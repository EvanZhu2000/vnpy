## NTH means only can run this script in non-trading hours
import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')     


from vnpy_portfoliostrategy.mysqlservice import MysqlService
import sys


def run(start_date_str, end_date_str):
    RECORD_FREQ = '1m'
    sqlservice = MysqlService()
    data = all_instruments(type='Future', market='cn', date=start_date_str)

    info = get_price(data['order_book_id'].loc[(data['order_book_id'].str.contains('99') == False) &
                    (data['order_book_id'].str.contains('88') == False)].dropna().unique().tolist(), 
                    start_date = start_date_str, 
                    end_date   = end_date_str, 
                    frequency=RECORD_FREQ)
    aaa = info.reset_index().set_index('order_book_id').join(data[['order_book_id', 'exchange','trading_code']].set_index('order_book_id')).reset_index().drop(['order_book_id'],axis=1).rename(columns={"total_turnover": "turnover", "open":"open_price","low":"low_price","high":"high_price","close":"close_price","trading_code":"symbol"})
    aaa['interval'] = RECORD_FREQ
    aaa = aaa[['symbol', 'exchange', 'datetime', 'interval', 'volume', 'turnover', 'open_interest', 'open_price', 'high_price', 'low_price', 'close_price']]
    sqlservice.insert_datafeed(aaa, ignore=True)

    sqlservice.close()


if __name__ == "__main__":
    if len(sys.argv) > 3 or len(sys.argv) < 3:
        raise Exception("Wrong number of input!!")
    start_date_str = sys.argv[1]
    end_date_str = sys.argv[2]
    run(start_date_str, end_date_str)
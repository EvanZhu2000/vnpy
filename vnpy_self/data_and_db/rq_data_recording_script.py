
from vnpy_portfoliostrategy.mysqlservice import MysqlService
from vnpy_self.ctp_setting import ctp_setting

from datetime import datetime, time
import sys
from time import sleep

import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

RECORD_INTERVAL = 5 # unit: minutes
RECORD_FREQ = '1m'

DAY_START = time(8, 45)
DAY_END = time(15, 35)

LUNCH_BREAK_START = time(11, 30 + RECORD_INTERVAL + 1)
LUNCH_BREAK_END = time(13, 0)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)


def check_trading_period():
    """"""
    current_time = datetime.now().time()

    trading = False
    if (
        (current_time >= DAY_START and current_time <= DAY_END)
        or (current_time >= NIGHT_START)
        or (current_time <= NIGHT_END)
    ):
        trading = True

    return trading


def is_lunch_break():
    current_time = datetime.now().time()
    return current_time >= LUNCH_BREAK_START and current_time <= LUNCH_BREAK_END



def run():
    sqlservice = MysqlService()
    tdy_str = datetime.today().strftime('%Y%m%d')
    data = all_instruments(type='Future', market='cn', date=tdy_str)

    while True:
        trading = check_trading_period()
        if not trading:
            print("Terminate data recording process")
            info = get_price(data['order_book_id'].loc[(data['order_book_id'].str.contains('99') == False) &
                    (data['order_book_id'].str.contains('88') == False)].dropna().unique().tolist(), 
                    start_date = tdy_str, 
                    end_date   = tdy_str, 
                    frequency=RECORD_FREQ)
            aaa = info.reset_index().set_index('order_book_id').join(data[['order_book_id', 'exchange','trading_code']].set_index('order_book_id')).reset_index().drop(['order_book_id'],axis=1).rename(columns={"total_turnover": "turnover", "open":"open_price","low":"low_price","high":"high_price","close":"close_price","trading_code":"symbol"})
            aaa['interval'] = RECORD_FREQ
            aaa = aaa[['symbol', 'exchange', 'datetime', 'interval', 'volume', 'turnover', 'open_interest', 'open_price', 'high_price', 'low_price', 'close_price']]
            sqlservice.insert_datafeed(aaa, ignore=True)

            sqlservice.close()
        else:
            if not is_lunch_break():
                info = get_price(data['order_book_id'].loc[(data['order_book_id'].str.contains('99') == False) &
                        (data['order_book_id'].str.contains('88') == False)].dropna().unique().tolist(), 
                        start_date = tdy_str, 
                        end_date   = tdy_str, 
                        frequency=RECORD_FREQ)
                aaa = info.swaplevel().sort_index().loc[info.swaplevel().sort_index().index.levels[0][-(RECORD_INTERVAL*2):]].reset_index().set_index('order_book_id').join(data[['order_book_id', 'exchange','trading_code']].set_index('order_book_id')).reset_index().drop(['order_book_id'],axis=1).rename(columns={"total_turnover": "turnover", "open":"open_price","low":"low_price","high":"high_price","close":"close_price","trading_code":"symbol"})
                aaa['interval'] = RECORD_FREQ
                aaa = aaa[['symbol', 'exchange', 'datetime', 'interval', 'volume', 'turnover', 'open_interest', 'open_price', 'high_price', 'low_price', 'close_price']]
                sqlservice.insert_datafeed(aaa, ignore=True)

        sleep(60*RECORD_INTERVAL)

if __name__ == "__main__":
    run()

        
        



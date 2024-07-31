from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_datarecorder.engine import  EVENT_RECORDER_LOG
from vnpy_ctp import CtpGateway
from vnpy_self.ctp_setting import ctp_setting

from vnpy_rqdata import *

from datetime import datetime, time
import sys
from time import sleep

import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','14261368Abc!')

# Chinese futures market trading period (day/night)
DAY_START = time(8, 45)
DAY_END = time(15, 35)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)

RECORD_INTERVAL = 5 # unit: minutes
RECORD_FREQ = '1m'


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



def run():
    """
    Running in the child process.
    """
    SETTINGS["log.file"] = True

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.init_engines()
    main_engine.add_gateway(CtpGateway)
    ps_engine = main_engine.add_app(PortfolioStrategyApp) 
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_RECORDER_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")
    sleep(60)

    tdy_str = datetime.today().strftime('%Y%m%d')
    data = all_instruments(type='Future', market='cn', date=tdy_str)

    while True:
        sleep(60*RECORD_INTERVAL)

        trading = check_trading_period()
        if not trading:
            print("Terminate data recording process")
            main_engine.close()
            sys.exit(0)
        else:
            info = get_price(data['order_book_id'].loc[(data['order_book_id'].str.contains('99') == False) &
                    (data['order_book_id'].str.contains('88') == False)].dropna().unique().tolist(), 
                    start_date = tdy_str, 
                    end_date   = tdy_str, 
                    frequency=RECORD_FREQ)
            aaa = info.swaplevel().sort_index().loc[info.swaplevel().sort_index().index.levels[0][-RECORD_INTERVAL:]].reset_index().set_index('order_book_id').join(data[['order_book_id', 'exchange','trading_code']].set_index('order_book_id')).reset_index().drop(['order_book_id'],axis=1).rename(columns={"total_turnover": "turnover", "open":"open_price","low":"low_price","high":"high_price","close":"close_price","trading_code":"symbol"})
            aaa['interval'] = RECORD_FREQ
            aaa = aaa[['symbol', 'exchange', 'datetime', 'interval', 'volume', 'turnover', 'open_interest', 'open_price', 'high_price', 'low_price', 'close_price']]
            ps_engine.dbservice.insert_datafeed(aaa)

if __name__ == "__main__":
    run()

        
        



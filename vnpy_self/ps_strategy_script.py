from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_portfoliostrategy.portfolio_rollover import RolloverTool

from datetime import datetime, time, timedelta
import sys
from time import sleep
import pandas as pd
import akshare as ak


SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True


ctp_setting = {
    "用户名": "224829",
    "密码": "evan@2024",
    "经纪商代码": "9999",
    "交易服务器": "tcp://180.168.146.187:10202",
    "行情服务器": "tcp://180.168.146.187:10212",
    "产品名称": "simnow_client_test",
    "授权编码": "0000000000000000",
    "产品信息": ""
}



# Chinese futures market trading period (day/night)
DAY_START = time(8, 45)
DAY_END = time(15, 0)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)

def determine_rollover_strategy1():
    result_df = pd.DataFrame(columns = ['symb_title', 'pre_roll', 'post_roll'])
    trading_instrument_dict = pd.read_excel(r'C:\\veighna_studio\\Lib\\site-packages\\vnpy_self\\strategy1_schedule.xlsx', sheet_name=None)
    pre_roll, post_roll = '',''
    for r in trading_instrument_dict['IH'][::-1].iterrows():  # assuming trading dates are sorted
        if datetime.strftime(datetime.today(),'%Y-%m-%d') < r[1]['date']:
            continue
        elif datetime.strftime(datetime.today(),'%Y-%m-%d') == r[1]['date']:
            # needs to rollover
            post_roll = r[1]['instrument']
        else:
            pre_roll = r[1]['instrument']
            break
        
    result_df.loc[result_df.shape[0]] = ['IH',pre_roll,post_roll]

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

def check_rollover_period():
    current_time = datetime.now().time()
    return current_time>=time(14, 50)

def run():
    SETTINGS["log.file"] = True
    rollover_today, old_series, new_series = determine_rollover_strategy1('IH')

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.init_engines()
    main_engine.add_gateway(CtpGateway)
    ps_engine = main_engine.add_app(PortfolioStrategyApp)   #  add portfolio strategy app
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_PORTFOLIO_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")

    ps_engine.init_engine()
    main_engine.write_log("ps策略初始化完成")
    ps_engine.add_strategy('Strategy1','Strategy1_IH',['IH2403.CFFEX','IH2406.CFFEX'],{})
    
    ps_engine.init_strategy('Strategy1_IH')
    main_engine.write_log("ps策略全部初始化")
    ps_engine.start_strategy('Strategy1_IH')
    main_engine.write_log("ps策略全部启动")
            
    while True:
        sleep(10)

        trading = check_trading_period()
        if (not trading) or (rollover_today and check_rollover_period()):
            ps_engine.stop_strategy('Strategy1_IH')
            if rollover_today:
                rt = RolloverTool(ps_engine=ps_engine, main_engine=main_engine)
                rt.init_symbols(old_series.tolist(), new_series.tolist())
                rt.roll_all()
            main_engine.close()
            sys.exit(0)

if __name__ == "__main__":
    run()

        
        

from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_portfoliostrategy.portfolio_rollover import RolloverTool
from vnpy_self.ctp_setting import ctp_setting
from vnpy_self.db_setting import db_setting

import re
from datetime import datetime, time, date
import sys
from time import sleep
import pandas as pd

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True

# Chinese futures market trading period (day/night)
DAY_START = time(8, 45)
DAY_END = time(15, 0)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)

def init_strategy1(sc_symbol, engine):
    trading_df = pd.DataFrame(columns = ['symb_title', 'pre_roll', 'post_roll'])
    trading_symbol_df = pd.read_sql_query(f"SELECT * FROM vnpy.trading_schedule where strategy = 'strategy1' and sc_symbol='{sc_symbol}' order by date desc", engine.mydb)
    pre_roll, post_roll = '',''
    if trading_symbol_df.shape[0] == 0:
        return trading_df
    for r in trading_symbol_df.iterrows():  # assuming trading dates are sorted
        if date.today() < r[1]['date']:
            continue
        elif date.today() == r[1]['date']:
            # needs to rollover
            post_roll = r[1]['symbol']
        else:
            pre_roll = r[1]['symbol']
            break
    
    if pre_roll == '' and post_roll != '':
        pre_roll = post_roll  
        post_roll = '' # This aims to make sure that pre_roll always points to current trading symbol (or first day trading symbol)
    trading_df.loc[trading_df.shape[0]] = [str(" ".join(re.findall("[a-zA-Z]+", sc_symbol))),pre_roll,post_roll]
    return trading_df

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

def parse_strategy(data, strategy_default_name):
    symb_title = data['symb_title']
    pre_roll = data['pre_roll'].split(',')
    post_roll = data['post_roll'].split(',')
    strategy_title = strategy_default_name + '_' + symb_title
    return pre_roll,post_roll,strategy_title

def run():
    SETTINGS["log.file"] = True
    
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
    sleep(10)

    ps_engine.init_engine()
    main_engine.write_log("ps策略初始化完成")
    
    
    trading_df = init_strategy1('IH_2', ps_engine)
    if trading_df.shape[0] == 0:
        print('dont have target symbol!!')
        return
    rollover_df = trading_df.loc[trading_df['post_roll']!='']
    
    
    for r in trading_df.iterrows():
        pre_roll,post_roll,strategy_title = parse_strategy(r[1], 'Strategy1')
        if strategy_title not in ps_engine.strategies.keys():
            ps_engine.add_strategy('Strategy1',strategy_title,pre_roll,{"boll_window": 10,"boll_dev": 2})
        ps_engine.init_strategy(strategy_title)
        main_engine.write_log("ps策略全部初始化")
        sleep(10)
        ps_engine.start_strategy(strategy_title)
        main_engine.write_log("ps策略全部启动")
            
    while True:
        sleep(10)
        
        if check_rollover_period():
            for r in rollover_df.iterrows():
                pre_roll,post_roll,strategy_title = parse_strategy(r[1], 'Strategy1')
                ps_engine.stop_strategy(strategy_title)
                ps_engine.remove_strategy(strategy_title)
                rt = RolloverTool(ps_engine=ps_engine, main_engine=main_engine)
                main_engine.write_log(f"Rolling from {pre_roll} to {post_roll}")
                rt.init_symbols(pre_roll, post_roll)
                rt.roll_all()
                
        if not check_trading_period():
            main_engine.write_log("ps策略全部close")
            ps_engine.stop_all_strategies()
            main_engine.close()
            sys.exit(0)

if __name__ == "__main__":
    run()

        
        

from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_self.ctp_setting import *
from vnpy.trader.constant import Direction
import json
from datetime import datetime, time, date
import sys
from time import sleep
import pandas as pd
import numpy as np
import signal
from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()
db.init_connection()

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True


def run(option:str):
    def signal_handler(signum, frame):
        main_engine.write_log("Received shutdown signal, closing ps strategy")
        ps_engine.stop_all_strategies()
        main_engine.close()
        
    SETTINGS["log.file"] = True
    
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.init_engines()
    main_engine.add_gateway(CtpGateway)
    ps_engine = main_engine.add_app(PortfolioStrategyApp)
    ctp_setting = ctp_map(option)
    main_engine.env = option
        
    main_engine.write_log("主引擎创建成功")
    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_PORTFOLIO_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")
    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")
    while not main_engine.get_gateway('CTP').td_api.contract_inited:
        sleep(1)
    ps_engine.init_engine()
    main_engine.write_log("ps策略初始化完成")
    
    strategy_title = 'strategy2'
    strategy_class_name = 'Strategy2'
    
    # === from db get stuff
    tmp = db.select('trading_schedule',today = datetime.today().date(), strategy = strategy_title)
    if tmp.empty:
        # today is the second day
        tmp = db.select('trading_schedule', date = datetime.today().date(), strategy = strategy_title)
        current_day = pd.to_datetime(tmp['today'].iloc[0]).strftime('%Y-%m-%d')
        settlement_dates_str = current_day+','+datetime.today().strftime('%Y-%m-%d')+':'+'Asia/Shanghai'
    else:
        # today is the first day
        current_day = datetime.today().strftime('%Y-%m-%d')
        next_day = pd.to_datetime(tmp['date'].iloc[0]).strftime('%Y-%m-%d')
        settlement_dates_str = current_day+','+next_day+':'+'Asia/Shanghai'
        
    # === current_day is supposingly when the script should be start running, ideally 21:00 every settlement date
    main_engine.write_log(f"settlement date starting day is {current_day}")
    
    to_trade_df = db.select('daily_rebalance_target',today = current_day, strategy = strategy_title)
    to_trade_df = pd.concat([pd.Series(to_trade_df['symbol'].values[0].split(',')),
                        pd.Series(to_trade_df['target'].values[0].split(','))],axis=1,keys=['symbol','target'])
    trading_hours = db.select('trading_hours')
    db.close()
    
    # ===== from CTP get pos_data
    omsEngine = main_engine.get_engine('oms')
    while(1):
        allpos = omsEngine.get_all_positions()
        if allpos is not None and len(allpos) != 0:
            break  
    abc = pd.DataFrame([x.__dict__ for x in allpos])
    qwe = pd.concat([abc['vt_symbol'],
                    abc['direction'].map({Direction.LONG:1,Direction.SHORT:-1}) * abc['volume']], axis=1)
    qwe = qwe.groupby(qwe['vt_symbol']).sum()
    pos_data = qwe.loc[qwe[0]!=0].sort_index().squeeze(axis=1).astype(int)
    main_engine.write_log(f"pos_data: {pos_data}")
    
    # ==== calculate positions
    to_trade_df['target'] = pd.to_numeric(to_trade_df['target'])
    ans = pos_data[['symbol','pos']].set_index('symbol').replace(0,np.nan).dropna().join(to_trade_df.drop_duplicates().set_index('symbol'),how='outer')
    ans = ans.replace(0,np.nan).dropna(how='all').replace(np.nan,0)
    
    vt_symbols = ans.index.values.tolist()
    settings = dict({'ans':json.dumps(ans.to_dict()),
                     'trading_hours':json.dumps(trading_hours[['symbol','trading_hours']].set_index('symbol').to_dict()['trading_hours']),
                     'settlement_dates_str':settlement_dates_str})
    
    # ===== start strategy
    if strategy_title in ps_engine.strategies.keys():
        ps_engine.stop_strategy(strategy_title)
        ps_engine.remove_strategy(strategy_title)
        ps_engine.add_strategy(strategy_class_name, strategy_title, vt_symbols, settings)
    else:
        ps_engine.add_strategy(strategy_class_name, strategy_title, vt_symbols, settings)
           
    ps_engine.init_strategy(strategy_title)
    while not ps_engine.strategies[strategy_title].inited:
        sleep(1)
    
    # ==== set pos_data
    strategy = ps_engine.strategies[strategy_title]
    for r in pos_data.iterrows():
        if r[1]['symbol'] in strategy.vt_symbols:
            strategy.set_pos(r[1]['symbol'], r[1]['pos'])
        
    main_engine.write_log("ps策略全部初始化")
    ps_engine.start_strategy(strategy_title)
    main_engine.write_log("ps策略全部启动")
    
    signal.signal(signal.SIGTERM, signal_handler)
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        run(current_environment)
    elif len(sys.argv) == 2:
        # the argument should be the CTP options
        run(sys.argv[1])
    else:
        raise Exception("Need to have zero or one argument for CTP options")


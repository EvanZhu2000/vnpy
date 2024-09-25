from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_portfoliostrategy.portfolio_rollover import RolloverTool
from vnpy_self.ctp_setting import ctp_setting

import json
from datetime import datetime, time, date
import sys
from time import sleep
import pandas as pd

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True

def strategy_running_period():
    """"""
    DAY_START = time(20, 45)
    DAY_END = time(15, 45)
    
    current_time = datetime.now().time()

    trading = False
    if (current_time >= DAY_START and current_time <= DAY_END):
        trading = True

    return trading

# NOTE: the rollover tool will get position from different strategies, so rollover can be anytime
def check_rollover_period():
    current_time = datetime.now().time()
    return current_time>=time(10, 50)

def run():
    SETTINGS["log.file"] = True
    
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.init_engines()
    main_engine.add_gateway(CtpGateway)
    ps_engine = main_engine.add_app(PortfolioStrategyApp)   
    main_engine.write_log("主引擎创建成功")
    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_PORTFOLIO_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")
    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")
    sleep(10)
    ps_engine.init_engine()
    main_engine.write_log("ps策略初始化完成")
    
    strategy_title = 'strategy2'
    strategy_class_name = 'Strategy2'
    
    tmp = ps_engine.dbservice.select('trading_schedule',today = datetime.today().date(), strategy = strategy_title)
    if tmp is None:
        tmp = ps_engine.dbservice.select('trading_schedule', date = datetime.today().date(), strategy = strategy_title)
        current_day = tmp['today']
    else:
        current_day = datetime.today()
    
    # fill positions and find target for today
    rebal_tar = ps_engine.dbservice.select('daily_rebalance_target',today = current_day, strategy = strategy_title)
    rebal_tar = pd.concat([pd.Series(rebal_tar['symbol'].values[0].split(',')),
                        pd.Series(rebal_tar['target'].values[0].split(','))],axis=1,keys=['symbol','target'])
    trading_schedule = ps_engine.dbservice.select('trading_schedule',today = current_day, strategy = strategy_title).set_index('id').drop_duplicates()
    previous_trading_schedule = ps_engine.dbservice.select('trading_schedule',date = current_day, strategy = strategy_title).set_index('id').drop_duplicates()
    trading_hours = ps_engine.dbservice.select('trading_hours',date = trading_schedule['date'].iloc[0])
    if trading_schedule.shape[0]!=1 or previous_trading_schedule.shape[0]!=1:
        raise Exception(f'Wrong trading schedule for {strategy_title}')
    to_trade_df = pd.concat([pd.Series(trading_schedule['symbol'].values[0].split(',')).str[:-4],
                             pd.Series(trading_schedule['symbol'].values[0].split(','))],axis=1,keys=['symbol','symb']
                            ).merge(trading_hours[['rqsymbol','symbol']],left_on='symb',right_on='rqsymbol',how='inner'
                                    ).merge(rebal_tar,left_on='symbol_x',right_on='symbol',how='inner'
                                            )[['symbol_y','target']]
    to_trade_df['target'] = pd.to_numeric(to_trade_df['target'])
    pos_data = ps_engine.get_pos(strategy_title)
    ans = pos_data[['symbol','pos']].set_index('symbol').join(to_trade_df.set_index('symbol_y'),how='outer')
    vt_symbols = ans.index.values.tolist()
    settings = dict({'tarpos':json.dumps(ans['target'].to_dict()),
                     'ans':json.dumps(ans.to_dict())})
    if strategy_title in ps_engine.strategies.keys():
        ps_engine.stop_strategy(strategy_title)
        ps_engine.remove_strategy(strategy_title)
        ps_engine.add_strategy(strategy_class_name, strategy_title, vt_symbols, settings)
    else:
        ps_engine.add_strategy(strategy_class_name, strategy_title, vt_symbols, settings)
        
    sleep(5)    
    ps_engine.init_strategy(strategy_title)
    main_engine.write_log("ps策略全部初始化")
    sleep(5)
    ps_engine.start_strategy(strategy_title)
    main_engine.write_log("ps策略全部启动")
    
    while True:
        sleep(5)     
        if not strategy_running_period():
            main_engine.write_log("ps策略全部close")
            ps_engine.stop_all_strategies()
            main_engine.close()
            sys.exit(0)

if __name__ == "__main__":
    run()

        
        

from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_self.ctp_setting import *
from vnpy_self.general import *
from vnpy.trader.constant import Direction
import json
from datetime import datetime, time, date
import sys
from time import sleep
import pandas as pd
import numpy as np

from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()
db.init_connection()

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True

def run(quickstart:str, option:str):
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
        
    # current_day is supposingly when the script should be start running, ideally 21:00 every settlement date
    main_engine.write_log(f"settlement date starting day is {current_day}")
    
    rebal_tar = db.select('daily_rebalance_target',today = current_day, strategy = strategy_title)
    rebal_tar = pd.concat([pd.Series(rebal_tar['symbol'].values[0].split(',')),
                        pd.Series(rebal_tar['target'].values[0].split(','))],axis=1,keys=['symbol','target'])
    trading_schedule = db.select('trading_schedule',today = current_day, strategy = strategy_title).drop_duplicates()
    previous_trading_schedule = db.select('trading_schedule',date = current_day, strategy = strategy_title).drop_duplicates()
    trading_hours = db.select('trading_hours',date = trading_schedule['date'].iloc[0])
    db.close()
    
    # ===== fill positions and find target for today
    if trading_schedule.shape[0]!=1 or previous_trading_schedule.shape[0]!=1:
        main_engine.write_exception(f'Wrong trading schedule for {strategy_title}')
    to_trade_df = pd.concat([pd.Series(trading_schedule['symbol'].values[0].split(',')).str[:-4],
                             pd.Series(trading_schedule['symbol'].values[0].split(','))],axis=1,keys=['symbol','symb']
                            ).merge(trading_hours[['rqsymbol','symbol']],left_on='symb',right_on='rqsymbol',how='inner'
                                    ).merge(rebal_tar,left_on='symbol_x',right_on='symbol',how='inner'
                                            )[['symbol_y','target']]
    to_trade_df['target'] = pd.to_numeric(to_trade_df['target'])
    pos_data = ps_engine.get_pos(strategy_title)
    ans = pos_data[['symbol','pos']].set_index('symbol').replace(0,np.nan).dropna().join(to_trade_df.drop_duplicates().set_index('symbol_y'),how='outer')
    ans = ans.replace(0,np.nan).dropna(how='all').replace(np.nan,0)
    
    vt_symbols = ans.index.values.tolist()
    settings = dict({'ans':json.dumps(ans.to_dict()),
                     'trading_hours':json.dumps(trading_hours[['symbol','trading_hours']].set_index('symbol').to_dict()['trading_hours']),
                     'settlement_dates_str':settlement_dates_str})
    
    # ===== Examine positions if necessary
    if quickstart == 'False':
        omsEngine = main_engine.get_engine('oms')
        while(1):
            allpos = omsEngine.get_all_positions()
            if allpos is not None and len(allpos) != 0:
                break;  
        abc = pd.DataFrame([x.__dict__ for x in allpos])
        qwe = pd.concat([abc['vt_symbol'],
                        abc['direction'].map({Direction.LONG:1,Direction.SHORT:-1}) * abc['volume']], axis=1)
        qwe = qwe.groupby(qwe['vt_symbol']).sum()
        qwe = qwe.loc[qwe[0]!=0].sort_index().squeeze(axis=1).astype(int)
        if not pos_data[['symbol','pos']].groupby('symbol').sum().query("pos!=0").sort_index().squeeze(axis=1).astype(int).equals(qwe):
            main_engine.write_exception("Wrong database positions record compared to CTP record")
        main_engine.write_log("Matching succeed: CTP and database")
    
    # ===== start strategy
    if strategy_title in ps_engine.strategies.keys():
        ps_engine.stop_strategy(strategy_title)
        ps_engine.remove_strategy(strategy_title)
        ps_engine.add_strategy(strategy_class_name, strategy_title, vt_symbols, settings)
    else:
        ps_engine.add_strategy(strategy_class_name, strategy_title, vt_symbols, settings)
           
    ps_engine.init_strategy(strategy_title)
    main_engine.write_log("ps策略全部初始化")
    while not ps_engine.strategies[strategy_title].inited:
        sleep(1)
    ps_engine.start_strategy(strategy_title)
    main_engine.write_log("ps策略全部启动")
    
    while True:
        sleep(60)
        if not check_trading_period_chinafutures():
            main_engine.write_log("ps策略全部close")
            ps_engine.stop_all_strategies()
            main_engine.close()

if __name__ == "__main__":
    quickstart = sys.argv[1] #(Should be either True or False)
    if len(sys.argv) == 2:
        run(quickstart, current_environment)
    elif len(sys.argv) == 3:
        # the second arguement should be the CTP options
        run(quickstart, sys.argv[2])
    else:
        raise Exception("Need to have one or two arguments, 1: whether to quickstart, 2: CTP options")

        
        

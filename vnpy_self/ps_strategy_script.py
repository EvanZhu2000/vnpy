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
import numpy as np
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

def determine_rollover_strategy1(strategy_symbol, rollover_days=7) -> bool:
    '''
    return whether strategy1 needs rollover, available to run before trading start
    '''
    futures_contract_info_cffex_df = None
    for i in range(0,15):
        try_this_date = (datetime.today() - timedelta(days = i)).strftime('%Y%m%d')
        try:
            futures_contract_info_cffex_df = ak.futures_contract_info_cffex(date=try_this_date)
            break
        except:
            continue
        
    if futures_contract_info_cffex_df is None:
        raise Exception('Cannot query future data')
    else:
        print(f"Query this date - {futures_contract_info_cffex_df['查询交易日'].loc[0].strftime('%Y%m%d')}")
        
    
    # Gets quarterly contracts info
    suitable_df = futures_contract_info_cffex_df.loc[(futures_contract_info_cffex_df['合约代码'].str.startswith(strategy_symbol)) & (futures_contract_info_cffex_df['合约月份'].astype(int) % 3 == 0)].sort_values(by=['最后交易日'])
    should_trade_series = suitable_df.loc[pd.to_datetime(suitable_df['最后交易日'])>datetime.today() + timedelta(days = rollover_days)].iloc[:2]['合约代码'].reset_index().drop(['index'],axis=1).squeeze()

    trading_instrument_df = pd.read_csv(r'C:\\veighna_studio\\Lib\\site-packages\\vnpy_self\\trading_instrument_list.csv')
    trading_series = pd.Series(trading_instrument_df.loc[trading_instrument_df['strategy_name'] == 'Strategy1']['instrument'].values[0].split(','), name='合约代码')
    
    rollover_flag = False
    if not trading_series.equals(should_trade_series):
        if should_trade_series.shape[0] == 2:
            if (should_trade_series.str.extract('(\d+)') <= trading_series.str.extract('(\d+)')).sum()[0] == 0:
                rollover_flag = True
            else:
                raise Exception('The rollover targer is earlier than the current contract')
        else:
            raise Exception('Do not have future data???')
    
    if rollover_flag:
        trading_instrument_df_copy = trading_instrument_df.copy()
        trading_instrument_df_copy.loc[trading_instrument_df_copy.loc[trading_instrument_df_copy['strategy_name'] == 'Strategy1'].index,'instrument'] = ','.join(should_trade_series.tolist())
        trading_instrument_df_copy.to_csv(r'C:\\veighna_studio\\Lib\\site-packages\\vnpy_self\\trading_instrument_list.csv')
    return rollover_flag, trading_series, should_trade_series

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

        
        

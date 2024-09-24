import sys
from time import sleep

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_self.ctp_setting import ctp_setting
import pandas as pd
from vnpy.trader.constant import Direction, Status
from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
SETTINGS["log.file"] = True

event_engine = EventEngine()
main_engine = MainEngine(event_engine)
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
main_engine.write_log("策略初始化完成")
tmp = main_engine.get_engine('oms').get_all_positions()

if len(tmp) == 0:
    raise Exception('position data not retrieved')
else:
    qwe = pd.DataFrame([x.__dict__ for x in tmp])
    ans = pd.concat([qwe['vt_symbol'],
           qwe['direction'].map({Direction.LONG:1,Direction.SHORT:-1}) * qwe['volume']], axis=1)
    
    strategy_title = 'strategy2'
    mysqlservice.delete_pos(strategy_title)
    for r in ans.iterrows():
        mysqlservice.update_pos(r[1]['vt_symbol'], strategy_title, r[1][0])
        
mysqlservice.close()
main_engine.close()
sys.exit(0)
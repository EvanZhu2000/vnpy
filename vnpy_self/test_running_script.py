import multiprocessing
import sys
from time import sleep
from datetime import datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG
from vnpy_self.ctp_setting import ctp_setting

def run_child():
    """
    Running in the child process.
    """
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
    strategy_name = "test1"
    if strategy_name not in ps_engine.strategies:
        ps_engine.add_strategy('SimpleBuyStrategy',strategy_name,['IH2406.CFFEX','IF2406.CFFEX'],{})
    ps_engine.init_strategy(strategy_name)
    sleep(10)   # Leave enough time to complete strategy initialization
    main_engine.write_log("策略初始化")
    print("策略初始化")

    ps_engine.start_strategy(strategy_name)
    main_engine.write_log("策略启动")

    sleep(120)
    ps_engine.stop_strategy(strategy_name)
    print("关闭子进程")
    main_engine.close()
    sys.exit(0)


if __name__ == "__main__":
    run_child()
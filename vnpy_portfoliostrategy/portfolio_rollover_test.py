from logging import INFO
import time
from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy_ctp import CtpGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG

from vnpy_portfoliostrategy.portfolio_rollover import RolloverTool


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

"""
Running in the child process.
"""
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
ps_engine.init_engine()
main_engine.write_log("ps策略初始化完成")

time.sleep(40)
# if 'test2' in ps_engine.strategies.keys():
#     ps_engine.remove_strategy('test2')
# ps_engine.add_strategy('SimpleBuyStrategy','test2',['IH2406.CFFEX','IH2409.CFFEX'],{})
ps_engine.init_strategy('test1')
main_engine.write_log("ps策略全部初始化")
ps_engine.start_strategy('test1')
main_engine.write_log("ps策略全部启动")

# time.sleep(80)
main_engine.write_log(ps_engine.strategies['test1'].target_data)
main_engine.write_log(ps_engine.strategies['test1'].pos_data)
ps_engine.stop_strategy('test1')


rt = RolloverTool(ps_engine=ps_engine, main_engine=main_engine)
rt.init('test1',['IH2406.CFFEX','IH2409.CFFEX'], ['IH2409.CFFEX','IH2412.CFFEX'])
rt.roll_all()

main_engine.close()
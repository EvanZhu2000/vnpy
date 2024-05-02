from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy_datarecorder.engine import RecorderEngine, EVENT_RECORDER_LOG
from vnpy_scripttrader.engine import ScriptEngine
from vnpy_ctp import CtpGateway
from vnpy.trader.constant import Product

from datetime import datetime, time
import sys
from time import sleep


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
    s_engine = main_engine.add_engine(ScriptEngine)
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_RECORDER_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")

    recorder=RecorderEngine(main_engine,event_engine)

    sleep(60)
    contract_df = s_engine.get_all_contracts(use_df=True)
    contract_list = contract_df.loc[(contract_df['product'] == Product.FUTURES) & (contract_df['symbol'].str.startswith('I'))]['vt_symbol'].values #获取中金所所有数据

    for contract in contract_list:
        print(f"Adding contract {contract}")
        recorder.add_bar_recording(contract)
        
    print(f"recorder.bar_recordings {recorder.bar_recordings}")
        
        
    while True:
        sleep(10)

        trading = check_trading_period()
        if not trading:
            print("Terminate data recording process")
            main_engine.close()
            sys.exit(0)

if __name__ == "__main__":
    run()

        
        

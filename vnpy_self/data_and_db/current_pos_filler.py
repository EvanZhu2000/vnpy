import sys
from time import sleep

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_self.ctp_setting import ctp_setting_uat, ctp_setting_live
import pandas as pd
from vnpy.trader.constant import Direction
from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
mysqlservice.init_connection()


def run(option:str):
    if option == 'uat':
        ctp_setting = ctp_setting_uat
    elif option == 'live':
        ctp_setting = ctp_setting_live
    else:
        raise Exception(f'Wrong option input {option}')
    
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    main_engine.write_log("主引擎创建成功")
    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")
    omsEngine = main_engine.get_engine('oms')
    while(1):
        tmp = omsEngine.get_all_positions()
        if tmp is not None and len(tmp) != 0:
            break;  
    tmp = omsEngine.get_all_positions()
    print(tmp)

    if len(tmp) == 0:
        raise Exception('position data not retrieved')
    else:
        qwe = pd.DataFrame([x.__dict__ for x in tmp])
        ans = pd.concat([qwe['vt_symbol'],
                        qwe['direction'].map({Direction.LONG:1,Direction.SHORT:-1}) * qwe['volume']], axis=1)
        ans = ans.groupby(ans['vt_symbol']).sum()
        
        strategy_title = 'strategy2'
        mysqlservice.delete_pos(strategy_title)
        mysqlservice.update_pos(strategy_title, ans.to_dict()[0])
        print("All finished")
    
    mysqlservice.close()
    main_engine.close()


if __name__ == "__main__":
    option = sys.argv[1]
    run(option)
    sys.exit(0)

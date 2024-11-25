import akshare as ak
from datetime import datetime
import pandas as pd
import sys
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_self.ctp_setting import *

from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
mysqlservice.init_connection()


def run(option:str):
    ctp_setting = ctp_map(option)
    pnl_directory = '//192.168.91.128/share_folder/Evan/PNL.xlsx'
    
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    main_engine.write_log("主引擎创建成功")
    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")
    omsEngine = main_engine.get_engine('oms')
    while(1):
        cur_bal = omsEngine.get_all_accounts()
        if cur_bal is not None and len(cur_bal) != 0:
            break;  
    cur_bal = omsEngine.get_all_accounts()[0].balance
    
    fx_spot_quote_df = ak.fx_spot_quote()
    hkdcny = fx_spot_quote_df.loc[fx_spot_quote_df['货币对'] == 'HKD/CNY'].eval('(买报价+卖报价)/2').values[0]
    
    ### read excel
    records = pd.read_excel(pnl_directory)
    last_day_records = records.sort_values('Date', ascending=True).iloc[-1]
    pre_bal, cum_pnl_percentage = last_day_records['Balance'], last_day_records['CUM_PNL%']
    
    pnl_CNY = cur_bal - pre_bal
    pnl_HKD = pnl_CNY * hkdcny
    pnl_percentage = pnl_CNY / pre_bal
    cum_pnl_percentage += pnl_percentage
    date = datetime.today().strftime("%YYYY-%mm-%dd")
    
    records.loc[len(records)] = [date, cur_bal, pnl_CNY, pnl_HKD, pnl_percentage, cum_pnl_percentage]

    ### write excel
    records.to_excel(pnl_directory)


if __name__ == "__main__":
    option = sys.argv[1]
    run(option)

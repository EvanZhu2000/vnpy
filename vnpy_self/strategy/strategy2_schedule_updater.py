from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
mysqlservice.init_connection()
import pandas as pd
import sys
import re

def run(today_date:str): 
    def get_rid_of_number(s):
        return re.sub(r'\d+', '', s)
    set1 = {'IF.CFFEX','IH.CFFEX','IC.CFFEX','IM.CFFEX','T.CFFEX','TS.CFFEX','TF.CFFEX','TL.CFFEX'}
    set2 = {'ag.SHFE','au.SHFE','sc.INE','ec.INE','cu.SHFE'}
    set3 = {'bc.INE', 'br.SHFE', 'CJ.CZCE', 'ec.INE', 'IH.CFFEX', 'nr.INE', 'pb.SHFE', 'sc.INE', 'SH.CZCE', 'sn.SHFE', 'TL.CFFEX', 'TS.CFFEX'} # self definition
    to_drop_list = list(set1|set2|set3)
    df = mysqlservice.select('trading_schedule', today = today_date, strategy='dom').iloc[0]
    potential_trading_series = pd.Series(df['symbol'].split(','))
    trading_list = potential_trading_series.loc[~potential_trading_series.apply(get_rid_of_number).isin(to_drop_list)].values
    mysqlservice.insert('trading_schedule',ignore=True,
                        today = today_date,date = df['date'], 
                        symbol = ','.join(trading_list), strategy='strategy2', sc_symbol='dom')
    mysqlservice.close()

if __name__ == "__main__":
    # The input today_date needs to be the real date at next settlement date start, in the format of YYYY-MM-DD
    today_date = sys.argv[1]
    
    import logging
    import os
    from datetime import datetime
    
    log_dir = f'/home/{os.getenv("APP_ENV", "uat")}/.vntrader/python_scripts_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(log_dir, f'python_script_{datetime.now().strftime("%Y%m%d")}.log')
    logging.basicConfig(filename=log_file, level=logging.ERROR, 
                        format='%(asctime)s %(levelname)s %(message)s')

    
    try:
        run(today_date)
        pass
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        sys.exit(1)
    




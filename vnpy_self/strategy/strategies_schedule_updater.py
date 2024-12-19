
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from vnpy_self.data_and_db.db_setting import *
from vnpy_self.alert_sender import send_email

from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()
db.init_connection()
import sys

def run(today_str:str): 
    next_day = db.select('trading_schedule',strategy='dom',today = today_str)['date'].values[0]
    strategies = db.select('strategies',date = today_str)
    strategies['date'] = next_day
    for s in strategies.drop('id',axis=1).iterrows():
        db.insert('strategies', ignore=True, **s[1].to_dict())
    db.close()
    
if __name__ == "__main__":
    # The input today_date needs to be the real date at next settlement date start, in the format of YYYY-MM-DD
    today_date = sys.argv[1]
        
    import logging
    import os
    from datetime import datetime
    
    log_filename = f'python_script_{datetime.now().strftime("%Y%m%d")}.log'
    log_path = f'/home/{os.getenv("APP_ENV", "uat")}/.vntrader/python_scripts_logs/{log_filename}'
    
    logging.basicConfig(
        filename=log_path,
        level=logging.ERROR,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    try:
        run(today_date)
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        sys.exit(1)
    

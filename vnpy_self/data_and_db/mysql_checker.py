
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from vnpy_self.data_and_db.db_setting import db_setting
from vnpy_self.alert_sender import send_email

from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()

# This is scheduled to check whether there are any mismatching in the database
if __name__ == "__main__":
    df = db.select('strategies',date = datetime.today().strftime('%Y-%m-%d'))
    strategy_list = df['strategy'].unique()
    wrong = []
    for s in strategy_list:
        t1 = db.get_pos(s)
        t2 = db.select('current_pos', 'order by datetime desc', strategy = s)[['symbol','pos']]
        if t1.shape!=t2.shape:
            wrong.append(s)
        elif (t2.sort_values('symbol')['pos'].values != t1.sort_values('symbol')['tar'].values).sum() !=0:
            wrong.append(s)
            
    if len(wrong)!=0:
        send_email('Mysql Check', f"{s} have mismatching current_pos")
    db.close()
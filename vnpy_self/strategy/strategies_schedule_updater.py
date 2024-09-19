
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from vnpy_self.data_and_db.db_setting import db_setting
from vnpy_self.alert_sender import send_email

from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()

# to update strategies after open
if __name__ == "__main__":
    today_str = datetime(2024,9,18).strftime('%Y-%m-%d')
    # today_str = datetime.today().strftime('%Y-%m-%d')
    next_day = db.select('trading_schedule',strategy='dom',today = today_str)['date'].values[0]
    strategies = db.select('strategies',date = today_str)
    strategies['date'] = next_day
    for s in strategies.drop('id',axis=1).iterrows():
        db.insert('strategies', ignore=True, **s[1].to_dict())
    db.close()
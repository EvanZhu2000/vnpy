
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from vnpy_self.data_and_db.db_setting import db_setting
from vnpy_self.alert_sender import send_email

from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()

# to update strategies after open
if __name__ == "__main__":
    next_day = db.select('trading_schedule',strategy='dom',today = datetime.today().strftime('%Y-%m-%d'))['date'].values[0]
    strategies = db.select('strategies',date = datetime.today().strftime('%Y-%m-%d'))
    strategies['date'] = next_day
    for s in strategies.iterrows():
        db.insert('strategies', **s[1].to_dict())
    db.close()
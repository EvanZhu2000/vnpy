
from datetime import datetime
from vnpy_self.data_and_db.db_setting import db_setting
import subprocess
from vnpy_self.data_and_db.db_setting import db_setting
from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()

if __name__ == "__main__":
    db.delete_datafeed()
    db.close()
    
# https://yarboroughtechnologies.com/how-to-automatically-backup-a-mysql-or-mariadb-server-with-mysqldump/ 
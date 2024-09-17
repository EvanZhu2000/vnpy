
from datetime import datetime
from dateutil.relativedelta import relativedelta
from vnpy_self.data_and_db.db_setting import db_setting

import mysql.connector
mydb = mysql.connector.connect(
  host= db_setting['host'],
  user= db_setting['user'],
  password= db_setting['password']
)
mycursor = mydb.cursor()

if __name__ == "__main__":
    mycursor.execute(f"DELETE FROM vnpy.dbbardata WHERE datetime < '{(datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d %H:%M:%S')}'")
    mydb.commit()
    mycursor.close()
    mydb.close()
    
# https://yarboroughtechnologies.com/how-to-automatically-backup-a-mysql-or-mariadb-server-with-mysqldump/ 
import pandas as pd

from vnpy_self.data_and_db.db_setting import db_setting
import mysql.connector
mydb = mysql.connector.connect(
  host= db_setting['host'],
  user= db_setting['user'],
  password= db_setting['password']
)
mycursor = mydb.cursor()

def run(strategy):
    orders = pd.read_sql_query(f"SELECT * FROM vnpy.strategy_order where strategy = '{strategy}' and order_status = 'Status.ALLTRADED';", mydb)
    print(orders.eval("price * (tar-pos)").groupby(orders['datetime'].dt.date).sum())

if __name__ == "__main__":
    run('strategy1')
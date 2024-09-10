import pandas as pd
import mysql.connector
from vnpy_self.data_and_db.db_setting import db_setting

class MysqlService():
    
    def __init__(self) -> None:
        self.mydb = mydb = mysql.connector.connect(
                host= db_setting['host'],
                user= db_setting['user'],
                password= db_setting['password']
            )
        self.mycursor = mydb.cursor()
        
    def close(self) -> None:
        self.mycursor.close()
        self.mydb.close()
    
    def dict_to_string(self,d):
        result = ""
        for key, value in d.items():
            result += f"{key} = '{value}' and "
        return result.rstrip(" and ")
        
    def insert(self,table,ignore=False,**kwargs) -> None:
        self.mycursor.execute(f"INSERT {'IGNORE' if ignore else ''} INTO `vnpy`.`{table}` "+\
                "(`" + "`, `".join(kwargs.keys()) + "`)" + \
                "VALUES" + "('"+ "', '".join(map(str,kwargs.values())) + "');")
        self.mydb.commit()
        
    def select(self,table,additional_query='',**where) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM `vnpy`.`{table}` {'where' if len(where)>0 else ''} {self.dict_to_string(where)}" + additional_query, self.mydb)
    
    def update_order_status(self, vt_orderid, order_status) -> pd.DataFrame:
        self.mycursor.execute(f"UPDATE`vnpy`.`strategy_order` SET order_status = '{order_status}' where vt_orderid = '{vt_orderid}';")
        self.mydb.commit()

    def insert_datafeed(self, data, ignore=False):
        query = f"INSERT {'IGNORE' if ignore else ''} INTO `vnpy`.`dbbardata` (`symbol`, `exchange`, `datetime`, `interval`, `volume`, `turnover`, `open_interest`, `open_price`, `high_price`, `low_price`, `close_price`) VALUES(%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s);"                                                         
        self.mycursor.executemany(query, list(map(tuple, data.values)))
        self.mydb.commit()
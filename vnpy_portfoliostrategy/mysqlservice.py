import pandas as pd
import mysql.connector
from vnpy.trader.setting import SETTINGS

class MysqlService():
    
    def __init__(self) -> None:
        self.mydb = mydb = mysql.connector.connect(
            host = SETTINGS["database.host"],
            user = SETTINGS["database.user"],
            password = SETTINGS["database.password"]
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
        
    def insert(self,table,**kwargs) -> None:
        self.mycursor.execute(f"INSERT INTO `vnpy`.`{table}` "+\
                "(`" + "`, `".join(kwargs.keys()) + "`)" + \
                "VALUES" + "('"+ "', '".join(map(str,kwargs.values())) + "');")
        self.mydb.commit()
        
    def select(self,table,additional_query,**where) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM `vnpy`.`{table}` where {self.dict_to_string(where)}" + additional_query, self.mydb)
    
    def update_order_status(self, vt_orderid, order_status) -> pd.DataFrame:
        self.mycursor.execute(f"UPDATE`vnpy`.`strategy_order` SET order_status = '{order_status}' where vt_orderid = '{vt_orderid}';")
        self.mydb.commit()
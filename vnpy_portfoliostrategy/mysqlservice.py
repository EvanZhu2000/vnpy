import pandas as pd
import mysql.connector
from dateutil.relativedelta import relativedelta
from datetime import datetime
from vnpy_self.data_and_db.db_setting import db_setting
import warnings
warnings.filterwarnings("ignore")

class MysqlService():
    
    def __init__(self) -> None:
        pass
        
    def init_connection(self) -> None:
        self.mydb = mysql.connector.connect(
                host= db_setting['host'],
                user= db_setting['user'],
                password= db_setting['password']
            )
        self.mycursor = self.mydb.cursor()
        
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
        
    def select(self,table,additional_query='',**where):
        return pd.read_sql_query(f"SELECT * FROM `vnpy`.`{table}` {'where' if len(where)>0 else ''} {self.dict_to_string(where)}" + additional_query, self.mydb)
        
    # def update(self, table, set_clause, **where) -> None:
    #     self.mycursor.execute(f"UPDATE `vnpy`.`{table}` SET {set_clause} {'where' if len(where)>0 else ''} {self.dict_to_string(where)};")
    
    '''
    to insert some data from rq to table_name
    '''
    def insert_rq(self, data: pd.DataFrame, table_name:str, ignore=False):
        s1 = ''
        s2 = ''
        for ind,l in enumerate(data.columns):
            if ind == 0:
                s1 += f'`{l}`'
                s2 += '%s'
            else:
                s1 += f',`{l}`'
                s2 += ',%s'
        query = f"INSERT {'IGNORE' if ignore else ''} INTO `vnpy`.`{table_name}` ({s1}) VALUES({s2});"                                                         
        self.mycursor.executemany(query, list(map(tuple, data.values)))
        self.mydb.commit()
        
    def delete_rq(self, table_name: str, dt_name, num_of_months=1):
        self.mycursor.execute(f"DELETE FROM vnpy.{table_name} WHERE {dt_name} < '{(datetime.now() - relativedelta(months=num_of_months)).strftime('%Y-%m-%d %H:%M:%S')}'")
        self.mydb.commit()
        
    # def get_pos(self, strategy_name):
    #     return pd.read_sql_query(f"select * from vnpy.strategy_order as sp join(SELECT symbol as latest_symbol, MAX(datetime) AS latest_timestamp, MAX(id) as max_id FROM vnpy.strategy_order where strategy = '{strategy_name}' and order_status = 'Status.ALLTRADED' GROUP BY symbol) as latest on sp.symbol = latest.latest_symbol and sp.datetime = latest.latest_timestamp and sp.id = latest.max_id;", self.mydb)[['symbol','tar']]
    
    def delete_pos(self, strategy):
        self.mycursor.execute(f"DELETE FROM vnpy.current_pos WHERE strategy='{strategy}';")
        self.mydb.commit()

    def update_pos(self, strategy_name, positions:dict[str, int]):
        self.delete_pos(strategy=strategy_name)
        for symbol,pos in positions.items():
            self.insert(table = 'current_pos', symbol = symbol, strategy = strategy_name, datetime = datetime.now(), pos = pos)
            
    
    # whenever there is an order update
    # def update_order_status(self, vt_orderid, order_status):
        # df = self.select('strategy_order', vt_orderid = vt_orderid)
        # df['order_status'] = order_status
        # df['datetime'] = datetime.now()
        # self.insert('strategy_order',ignore=True,**df.drop(['id'],axis=1).iloc[0].to_dict())

import pandas as pd
import mysql.connector
from dateutil.relativedelta import relativedelta
from datetime import datetime
from vnpy_self.data_and_db.db_setting import *
import warnings
from vnpy_portfoliostrategy.helperclass import *
warnings.filterwarnings("ignore")

class MysqlService():
    
    def __init__(self, option=None):
        if option is None:
            self.db_config = get_db_settings()
        else:
            self.db_config = db_map(option)

    def init_connection(self):
        self.mydb = mysql.connector.connect(
            host=self.db_config['host'],
            user=self.db_config['user'],
            password=self.db_config['password']
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
            
    def insert(self, table, ignore=False, replace=False, **kwargs) -> None:
        """
        Insert data into table with options for IGNORE or REPLACE behavior
        
        Args:
            table (str): Table name
            ignore (bool): Use INSERT IGNORE if True
            replace (bool): Use INSERT ... ON DUPLICATE KEY UPDATE if True
            **kwargs: Column names and values to insert
        """
        if ignore and replace:
            raise ValueError("Cannot use both ignore and replace options simultaneously")
        
        if replace:
            # Create the base INSERT part
            query = f"INSERT INTO `vnpy`.`{table}` " + \
                    "(`" + "`, `".join(kwargs.keys()) + "`)" + \
                    "VALUES" + "('"+ "', '".join(map(str,kwargs.values())) + "')"
            
            # Add ON DUPLICATE KEY UPDATE part for all fields except the primary key (symbol)
            update_fields = [key for key in kwargs.keys() if key != 'symbol']
            update_pairs = [f"`{key}` = VALUES(`{key}`)" for key in update_fields]
            query += " ON DUPLICATE KEY UPDATE " + ", ".join(update_pairs) + ";"
        else:
            # Original INSERT or INSERT IGNORE behavior
            query = f"INSERT {'IGNORE' if ignore else ''} INTO `vnpy`.`{table}` " + \
                    "(`" + "`, `".join(kwargs.keys()) + "`)" + \
                    "VALUES" + "('"+ "', '".join(map(str,kwargs.values())) + "');"
        
        self.mycursor.execute(query)
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


    def update_pos(self,  positions:dict[str, PositionInfo]):
        self.mycursor.execute(f"DELETE FROM vnpy.current_pos;")
        self.mydb.commit()
        for symbol,pos in positions.items():
            self.insert(table = 'current_pos', symbol = symbol, datetime = datetime.now(), pos = pos.__str__())
            
    
    # whenever there is an order update
    # def update_order_status(self, vt_orderid, order_status):
        # df = self.select('strategy_order', vt_orderid = vt_orderid)
        # df['order_status'] = order_status
        # df['datetime'] = datetime.now()
        # self.insert('strategy_order',ignore=True,**df.drop(['id'],axis=1).iloc[0].to_dict())

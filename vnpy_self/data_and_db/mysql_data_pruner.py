from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()
db.init_connection()

if __name__ == "__main__":
    db.delete_rq('dbbardata', 'datetime', num_of_months=12)
    db.delete_rq('dbdominant', 'date', num_of_months=12)
    db.delete_rq('dbmemberrank', 'trading_date', num_of_months=12)
    db.close()
    
# https://yarboroughtechnologies.com/how-to-automatically-backup-a-mysql-or-mariadb-server-with-mysqldump/ 
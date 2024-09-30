from vnpy_portfoliostrategy.mysqlservice import MysqlService
db = MysqlService()

if __name__ == "__main__":
    db.delete_datafeed(num_of_months=12)
    db.close()
    
# https://yarboroughtechnologies.com/how-to-automatically-backup-a-mysql-or-mariadb-server-with-mysqldump/ 
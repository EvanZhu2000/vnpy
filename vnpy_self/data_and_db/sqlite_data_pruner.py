import sqlite3
from datetime import datetime
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    conn = sqlite3.connect(r'c:\Users\Chris\.vntrader\database.db')
    # delete data 1 year from now
    cursor = conn.execute(f"DELETE FROM 'dbbardata' WHERE datetime < '{(datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d %H:%M:%S')}'")
    for row in cursor:
        print(row)
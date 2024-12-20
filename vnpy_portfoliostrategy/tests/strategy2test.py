from datetime import datetime
import json
import os
import pandas as pd
from importlib import reload
import vnpy_portfoliostrategy
reload(vnpy_portfoliostrategy)
from vnpy_portfoliostrategy import BacktestingEngine
from vnpy.trader.constant import Interval
import vnpy_portfoliostrategy.strategies.strategy2 as stg
reload(stg)
from vnpy_portfoliostrategy.strategies.strategy2 import Strategy2


if __name__ == "__main__":
    vt_symbols=["fu2501.SHFE"]
    current_dir = os.path.dirname(__file__)
    csv_file_path = os.path.join(current_dir, 'testfiles', 'stg2ticks.csv')
    
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.TICK,
        start=datetime(2024, 11, 14),
        end=datetime(2024, 11, 14),
        sizes={
            vt_symbols[0]: 10
        },
        capital=10000000,
        file_path=csv_file_path
    )
    trading_hours = {"fu.SHFE":'21:01-23:00,09:01-10:15,10:31-11:30,13:31-15:00'}
    ans = pd.DataFrame([[10,0]],
                        index=pd.Index(['fu2501.SHFE'],name='symbol'),
                        columns = ['target','pos'])
    settings = dict({'ans':json.dumps(ans.to_dict()),
                     'trading_hours':json.dumps(trading_hours),
                     'settlement_dates_str':'2024-11-14,2024-11-15:Asia/Shanghai'})
    engine.add_strategy(Strategy2, settings)
    engine.init_strategy()
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    trades = engine.get_all_trades(use_df=True)
    orders = pd.DataFrame([x.__dict__ for x in engine.get_all_orders()])
    
    if engine.strategy.trades:
        rows = [
            (date, tr.datetime, tr.vt_symbol, tr.vt_orderid, tr.direction, tr.offset, tr.price, tr.volume)
            for date, trades in engine.strategy.trades.items()
            for tr in trades
        ]
        trade_records = pd.DataFrame(rows, columns=['signal_datetime', 'datetime','vt_symbol', 'vt_orderid','direction','offset','price', 'volume'])
        trade_records[['direction','offset']] = trade_records[['direction','offset']].astype(str)
        print(trade_records)
        # print(pd.DataFrame(engine.strategy.trades))
        # print(pd.DataFrame([x.__dict__ for x in list(engine.strategy.trades.values())[0]],
        #                     index = [engine.strategy.trades.keys()]*len(list(engine.strategy.trades.values())[0])))
        
        
    # if not trades.empty:
    #     print(trades[['datetime','vt_symbol', 'vt_orderid','direction','offset','price', 'volume']])
    # if not orders.empty:
    #     print(orders[['datetime','vt_symbol', 'vt_orderid','direction','offset','price', 'volume','type', 'traded', 'status']])
    
    # print(','.join(engine.logs))
    
    # engine.calculate_statistics()
    # engine.show_chart()
    # engine.get_all_trades(use_df=True).to_csv(r'C:\\Users\\Chris\\Desktop\\Evan\\trades.csv')
from datetime import datetime
import json
import os
import pandas as pd
from vnpy_portfoliostrategy import BacktestingEngine
from vnpy.trader.constant import Interval
from vnpy_portfoliostrategy.strategies.strategy2 import Strategy2
import unittest


class TestStrategy2(unittest.TestCase):
    def setUp(self):
        self.vt_symbols = ["fu2501.SHFE"]
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.csv_file_path = os.path.join(current_dir, 'vnpy_portfoliostrategy', 'tests', 'testfiles', 'stg2ticks.csv')
        
        self.engine = BacktestingEngine()
        self.engine.set_parameters(
            vt_symbols=self.vt_symbols,
            interval=Interval.TICK,
            start=datetime(2024, 11, 14),
            end=datetime(2024, 11, 14),
            sizes={
                "fu2501.SHFE": 10
            },
            capital=10000000,
            file_path=self.csv_file_path
        )
        
        trading_hours = {"fu.SHFE":'21:01-23:00,09:01-10:15,10:31-11:30,13:31-15:00'}
        ans = pd.DataFrame([[10,0]],
                          index=pd.Index(['fu2501.SHFE'],name='symbol'),
                          columns = ['target','pos'])
        self.settings = dict({
            'ans':json.dumps(ans.to_dict()),
            'trading_hours':json.dumps(trading_hours),
            'settlement_dates_str':'2024-11-14,2024-11-15:Asia/Shanghai'
        })
        self.engine.add_strategy(Strategy2, self.settings)
        self.engine.init_strategy()
        self.engine.load_data()
        self.engine.run_backtesting()




    def test_strategy_execution(self):
        # Test if trades were generated
        # Get actual results
        trades = self.engine.get_all_trades(use_df=True).reset_index(drop=True).astype(str)
        orders = pd.DataFrame([x.__dict__ for x in self.engine.get_all_orders()]).reset_index(drop=True).astype(str)
        
        # Create trade records if they exist
        trade_records = pd.DataFrame()
        if self.engine.strategy.trades:
            rows = [
                (date, tr.datetime, tr.vt_symbol, tr.vt_orderid, tr.direction, tr.offset, tr.price, tr.volume)
                for date, trades in self.engine.strategy.trades.items()
                for tr in trades
            ]
            trade_records = pd.DataFrame(rows, columns=['signal_datetime', 'datetime','vt_symbol', 'vt_orderid','direction','offset','price', 'volume'])
            trade_records = trade_records.reset_index(drop=True).astype(str)

        # Load expected results from expect folder
        current_dir = os.path.dirname(os.path.dirname(__file__))
        expect_dir = os.path.join(current_dir, 'tests','expect')

        expected_trades = pd.read_csv(os.path.join(expect_dir, 'trades.csv')).astype(str)   
        expected_orders = pd.read_csv(os.path.join(expect_dir, 'orders.csv')).astype(str)
        expected_trade_records = pd.read_csv(os.path.join(expect_dir, 'trade_records.csv')).astype(str)
        
        # Compare actual and expected results
        try:
            pd.testing.assert_frame_equal(trades, expected_trades, check_dtype=False)
        except AssertionError as e:
            print("\nTrades comparison failed:")
            print(e)
            raise

        try:
            pd.testing.assert_frame_equal(orders[expected_orders.columns], expected_orders, check_dtype=False)
        except AssertionError as e:
            print("\nOrders comparison failed:")
            print(e)
            raise

        try:
            pd.testing.assert_frame_equal(trade_records, expected_trade_records, check_dtype=False)
        except AssertionError as e:
            print("\nTrade records comparison failed:")
            print(e)
            raise


if __name__ == "__main__":
    unittest.main()
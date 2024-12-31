from datetime import datetime
import json
import os
import pandas as pd
from vnpy_portfoliostrategy import BacktestingEngine
from vnpy.trader.constant import Interval
from vnpy_portfoliostrategy.strategies.strategy2 import Strategy2
import unittest
import pytz


class TestStrategy2(unittest.TestCase):
    def setUp(self):
        pass

    def test_alert_too_long(self):
        self.vt_symbols = ["fu2501.SHFE","cs2501.DCE"]
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.csv_file_path = os.path.join(current_dir, 'vnpy_portfoliostrategy', 'tests', 'testfiles', 'alert1.csv')
        
        self.engine = BacktestingEngine()
        self.engine.set_parameters(
            vt_symbols=self.vt_symbols,
            interval=Interval.TICK,
            start=datetime(2024, 11, 14),
            end=datetime(2024, 11, 14),
            sizes={
                "fu2501.SHFE": 10,
                "cs2501.DCE": 10
            },
            capital=10000000,
            file_path=self.csv_file_path
        )
        self.engine.starting_time = datetime(2024, 11, 14, 20, 55, 0).astimezone(pytz.timezone('Asia/Shanghai'))
        trading_hours = {"fu.SHFE":'21:01-23:00,09:01-10:15,10:31-11:30,13:31-15:00',
                         "cs.DCE":'21:01-23:00,09:01-10:15,10:31-11:30,13:31-15:00'}
        ans = pd.DataFrame([[100,0],[0,50]],
                          index=pd.Index(['fu2501.SHFE','cs2501.DCE'],name='symbol'),
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
           # Convert logs to DataFrame
        logs = self.engine.logs
        log_data = []
        for log in logs:
            timestamp, message = log.split('\t')
            log_data.append({
                'timestamp': timestamp.strip(),
                'message': message.strip()
            })
            # print(timestamp.strip(), ' ', message.strip())
        log_df = pd.DataFrame(log_data)
        
        try:
            pd.testing.assert_frame_equal(log_df.loc[log_df['message'].str.contains('Too long since last tick')].set_index('timestamp'), 
                                          pd.read_csv(os.path.join(current_dir, 'vnpy_portfoliostrategy','tests','expect', 'Too_long_since_last_tick.csv')).astype(str).set_index('timestamp'), 
                                          check_dtype=False)
        except AssertionError as e:
            print("\ntest_alert_too_long failed:")
            print(e)
            raise

    # def test_alert_first_tick(self):
    #     self.vt_symbols = ["fu2501.SHFE","cs2501.DCE"]
    #     current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    #     self.csv_file_path = os.path.join(current_dir, 'vnpy_portfoliostrategy', 'tests', 'testfiles', 'alert2.csv')
        
    #     self.engine = BacktestingEngine()
    #     self.engine.set_parameters(
    #         vt_symbols=self.vt_symbols,
    #         interval=Interval.TICK,
    #         start=datetime(2024, 11, 14),
    #         end=datetime(2024, 11, 14),
    #         sizes={
    #             "fu2501.SHFE": 10,
    #             "cs2501.DCE": 10
    #         },
    #         capital=10000000,
    #         file_path=self.csv_file_path
    #     )
    #     self.engine.starting_time = datetime(2024, 11, 14, 20, 55, 0).astimezone(pytz.timezone('Asia/Shanghai'))
    #     trading_hours = {"fu.SHFE":'21:01-23:00,09:01-10:15,10:31-11:30,13:31-15:00',
    #                      "cs.DCE":'21:01-23:00,09:01-10:15,10:31-11:30,13:31-15:00'}
    #     ans = pd.DataFrame([[100,0],[0,50]],
    #                       index=pd.Index(['fu2501.SHFE','cs2501.DCE'],name='symbol'),
    #                       columns = ['target','pos'])
    #     self.settings = dict({
    #         'ans':json.dumps(ans.to_dict()),
    #         'trading_hours':json.dumps(trading_hours),
    #         'settlement_dates_str':'2024-11-14,2024-11-15:Asia/Shanghai'
    #     })
    #     self.engine.add_strategy(Strategy2, self.settings)
    #     self.engine.init_strategy()
    #     self.engine.load_data()
    #     self.engine.run_backtesting()
    #        # Convert logs to DataFrame
    #     logs = self.engine.logs
    #     log_data = []
    #     for log in logs:
    #         timestamp, message = log.split('\t')
    #         log_data.append({
    #             'timestamp': timestamp.strip(),
    #             'message': message.strip()
    #         })
    #         print(timestamp.strip(), ' ', message.strip())
    #     log_df = pd.DataFrame(log_data)
        
    #     try:
    #         pd.testing.assert_frame_equal(log_df.loc[log_df['message'].str.contains("didn't receive any ticks")].set_index('timestamp'), 
    #                                       pd.read_csv(os.path.join(current_dir, 'vnpy_portfoliostrategy','tests','expect', 'First_tick_alert.csv')).astype(str).set_index('timestamp'), 
    #                                       check_dtype=False)
    #     except AssertionError as e:
    #         print("\ntest_alert_too_long failed:")
    #         print(e)
    #         raise


if __name__ == "__main__":
    unittest.main().test_alert_first_tick()
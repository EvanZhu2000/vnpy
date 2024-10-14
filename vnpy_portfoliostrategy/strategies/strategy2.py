
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy_portfoliostrategy.booldict import BoolDict
from vnpy.trader.object import TickData
import pandas as pd
from datetime import datetime, time, timedelta
import json


class Strategy2(StrategyTemplate):
   
    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.bool_dict = BoolDict()
        for symb in vt_symbols:
            self.bool_dict.set(symb, False)
        if 'ans' in setting:
            tarpos = json.loads(setting['ans'])['target']
            curpos = json.loads(setting['ans'])['pos']
            self.write_log(f"curpos {curpos}")
            self.write_log(f"tarpos {tarpos}")
            for symb,tar in tarpos.items():
                self.set_target(symb, tar)
        if 'trading_hours' in setting:
            self.trading_hours = json.loads(setting['trading_hours'])
    
    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")
        self.put_event()
        
    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")
        self.put_event()
        
    def on_stop(self) -> None:
        """策略停止回调"""
        super().on_stop()
        self.write_log("策略停止")
        self.put_event()
        
    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        if self.bool_dict.all_true():
            self.write_log("All have rebalanced. Stop the strategy now")
            self.strategy_engine.stop_strategy(self.strategy_name)
            return
        
        if tick.vt_symbol not in self.trading_hours.keys():
            self.write_log(f"No trading hours provided for {tick.vt_symbol}")
            self.strategy_engine.stop_strategy(self.strategy_name)
            return
        else:
            continuous_trading_intervals = self.trading_hours[tick.vt_symbol]
            if not self.is_time_in_intervals(tick.datetime.time(), continuous_trading_intervals):
                # Then this tick is not a continuous trading tick
                return
        
        if (self.get_target(tick.vt_symbol) != self.get_pos(tick.vt_symbol)):
            if (not self.symbol_is_active[tick.vt_symbol]):
                bp,ap = self.get_retry_price(tick)
                self.rebalance(tick.vt_symbol, bp, ap, net=False, strategy='strategy2',intention='rebalance')
        else:
            self.bool_dict.set(tick.vt_symbol, True)
    
    # check trading hours, may not belong to here 
    def is_time_in_intervals(input_time, intervals):
        # Split intervals and check each one
        for interval in intervals.split(','):
            start_str, end_str = interval.split('-')
            
            # Parse start and end times
            start_time = datetime.strptime(start_str, '%H:%M').time()
            end_time = datetime.strptime(end_str, '%H:%M').time()
            
            # Adjust start_time to be one minute earlier
            adjusted_start_time = (datetime.combine(datetime.today(), start_time) - timedelta(minutes=1)).time()
            
            # Check if the input time is within the adjusted interval
            if adjusted_start_time <= input_time <= end_time:
                return True
                
        return False
            
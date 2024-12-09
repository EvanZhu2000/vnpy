
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy_portfoliostrategy.helperclass import *
from vnpy.trader.object import TickData
from datetime import datetime, timedelta
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
        # self.tick_tracker = BoolDict(vt_symbols)
        # self.time_since_first_tick = timedelta(minutes=1)
        self.write_log(f"vt_symbols {vt_symbols}")
        
        if 'ans' in setting:
            tarpos = json.loads(setting['ans'])['target']
            curpos = json.loads(setting['ans'])['pos']
            self.write_log(f"curpos {self.nonzero_dict(curpos)}")
            self.write_log(f"tarpos {self.nonzero_dict(tarpos)}")
            for symb,tar in tarpos.items():
                self.set_target(symb, tar)
                
        if 'trading_hours' in setting:
            self.trading_hours = json.loads(setting['trading_hours'])
            target_time_collection = dict()
            for symb,th in self.trading_hours.items():
                target_time_collection[symb] = self.get_open_time(th)
            
        self.rebal_tracker = BoolDict(vt_symbols, target_time_collection)
    
    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")
        self.put_event()
        
    def on_start(self) -> None:
        """策略启动回调"""
        super().on_start()
        self.write_log("策略启动")
        self.put_event()
        
    def on_stop(self) -> None:
        """策略停止回调"""
        super().on_stop()
        self.write_log("策略停止")
        self.put_event()
        
    def on_tick(self, tick: TickData) -> None:
        if not self.trading or not super().on_tick(tick):
            return
        
        if self.rebal_tracker.all_true():
            self.strategy_engine.stop_strategy(self.strategy_name,
                                               f"All have rebalanced. Stop the strategy {self.strategy_name} now",
                                               f"{self.strategy_name}_success_{self.strategy_engine.main_engine.env}")
            return
        
        if tick.datetime - self.rebal_tracker.target_time_dict[tick.vt_symbol] > self.time_since_last_tick:
            # for this strategy, stop the ticker if it is not subscripable
            self.symbol_status[tick.vt_symbol].stop_late_tick_rebal = True
            self.rebal_tracker.true_count += 1
            
            ### Stop the entire strategy
            # self.strategy_engine.stop_strategy(self.strategy_name,
            #                                 f"Cannot receive ticks after {self.time_since_first_tick} for instruments {self.tick_tracker.get_false_keys()},   starting {self.starting_time}",
            #                                 f"{self.strategy_name}_fail_{self.strategy_engine.main_engine.env}")
            return

        if (self.get_target(tick.vt_symbol) != self.get_pos(tick.vt_symbol)):
            if (not self.symbol_status[tick.vt_symbol].is_active and not self.symbol_status[tick.vt_symbol].is_stop()):
                bp,ap = self.get_retry_price(tick)
                self.rebalance(tick.vt_symbol, bp, ap, net=False, strategy='strategy2',intention='rebalance')
        else:
            self.rebal_tracker.set(tick.vt_symbol, True)
    

            

from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy_portfoliostrategy.booldict import BoolDict
from vnpy.trader.object import TickData
import pandas as pd
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
            tarpos = json.loads(setting['ans']['target'])
            self.write_log(f"curpos {setting['ans']['pos']}")
            self.write_log(f"tarpos {tarpos}")
            for symb,tar in tarpos.items():
                self.set_target(symb, tar)
    
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
        self.write_log("策略停止")
        self.put_event()
        
    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        if self.bool_dict.all_true():
            self.write_log("All have rebalanced. Stop the strategy now")
            self.strategy_engine.stop_strategy(self.strategy_name)
            return
        
        if (self.get_target(tick.vt_symbol) != self.get_pos(tick.vt_symbol)):
            if (not self.symbol_is_active[tick.vt_symbol]):
                self.rebalance(tick.vt_symbol, tick.ask_price_1, tick.bid_price_1, net=False, strategy='strategy2',intention='rebalance')
        else:
            self.bool_dict.set(tick.vt_symbol, True)
            
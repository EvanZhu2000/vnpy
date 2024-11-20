
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy_portfoliostrategy.booldict import BoolDict
from vnpy.trader.object import TickData
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
        self.write_log(f"vt_symbols {vt_symbols}")
        for symb in vt_symbols:
            self.bool_dict.set(symb, False)
        if 'ans' in setting:
            tarpos = json.loads(setting['ans'])['target']
            curpos = json.loads(setting['ans'])['pos']
            self.write_log(f"curpos {self.nonzero_dict(curpos)}")
            self.write_log(f"tarpos {self.nonzero_dict(tarpos)}")
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
        if not self.trading or not super().on_tick(tick):
            return
        
        if self.bool_dict.all_true():
            self.strategy_engine.stop_strategy(self.strategy_name,
                                               f"All have rebalanced. Stop the strategy {self.strategy_name} now")
            return
        
        if (self.get_target(tick.vt_symbol) != self.get_pos(tick.vt_symbol)):
            if (not self.symbol_status[tick.vt_symbol].is_active and not self.symbol_status[tick.vt_symbol].is_stop()):
                bp,ap = self.get_retry_price(tick)
                self.rebalance(tick.vt_symbol, bp, ap, net=False, strategy='strategy2',intention='rebalance')
        else:
            self.bool_dict.set(tick.vt_symbol, True)
    

            
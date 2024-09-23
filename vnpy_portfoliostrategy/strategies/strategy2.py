from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    BarData,
    OrderType
)
import pandas as pd

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
        self.write_log(f"vt_symbols {vt_symbols}")
        if 'tarpos' in setting:
            tarpos = pd.Series(setting['tarpos'].split(',') ).astype(int).values
            self.write_log(f"tarpos {tarpos}")
            for i in range(len(vt_symbols)):
                symb = vt_symbols[i]
                if 'tarpos' in setting:
                    tar = tarpos[i]
                    self.set_target(symb, tar)

    
    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")
        self.put_event()
        
    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")
        for s in self.vt_symbols:
            print(f"pos - {self.get_pos(s)}; tar - {self.get_target(s)}")
        self.put_event()
        
    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log("策略停止")
        self.put_event()
        
    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        if (self.get_target(tick.vt_symbol) != self.get_pos(tick.vt_symbol)) and (not self.symbol_is_active[tick.vt_symbol]):
            self.rebalance(tick.vt_symbol, tick.ask_price_1, tick.bid_price_1, 'strategy2','rebalance')
from datetime import datetime

import numpy as np

from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import TickData, BarData
from vnpy.trader.constant import Direction

from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine


class SimpleBuyStrategy(StrategyTemplate):
    """配对交易策略"""
    
    tick_add = 0
    fixed_size = 1
    
    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)

        self.bgs: dict[str, BarGenerator] = {}
        self.last_tick_time: datetime = None
        self.leg1_symbol, self.leg2_symbol = vt_symbols

        def on_bar(bar: BarData):
            """"""
            pass

        for vt_symbol in self.vt_symbols:
            self.bgs[vt_symbol] = BarGenerator(on_bar)

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.load_bars(10)

    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        if (
            self.last_tick_time
            and self.last_tick_time.minute != tick.datetime.minute
        ):
            bars = {}
            for vt_symbol, bg in self.bgs.items():
                bars[vt_symbol] = bg.generate()
            self.on_bars(bars)

        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)

        self.last_tick_time = tick.datetime
        
        if self.target_pos_mismatch():
            self.rebalance_portfolio(bars)
            
    def target_pos_mismatch(self) -> bool:
        flag = False
        for vt_symbol in self.vt_symbols:
            if self.get_pos(vt_symbol) != self.get_target(vt_symbol):
                flag = True
                break
        return flag

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        leg1_bar = bars.get(self.leg1_symbol, None)
        leg2_bar = bars.get(self.leg2_symbol, None)
        if not leg1_bar or not leg2_bar:
            return

        self.set_target(self.leg1_symbol, -self.fixed_size)
        self.set_target(self.leg2_symbol, self.fixed_size)
        self.rebalance_portfolio(bars)
        
        # not sure whether necessary
        self.put_event()
        

    def calculate_price(self, vt_symbol: str, direction: Direction, reference: float) -> float:
        """计算调仓委托价格（支持按需重载实现）"""
        pricetick: float = self.get_pricetick(vt_symbol)

        if direction == Direction.LONG:
            price: float = reference + self.tick_add * pricetick
        else:
            price: float = reference - self.tick_add * pricetick

        return price

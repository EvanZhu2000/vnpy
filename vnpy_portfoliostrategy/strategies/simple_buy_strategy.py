from datetime import datetime

import numpy as np

from vnpy.trader.utility import BarGenerator, ArrayManager, Interval
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    BarData,
    OrderType
)
from vnpy.trader.constant import Direction, Status

from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy_portfoliostrategy.utility import PortfolioBarGenerator

# From the log we can see it is using only 1 order id
class SimpleBuyStrategy(StrategyTemplate):
    """配对交易策略"""
    
    tick_add = 0
    fixed_size = 1
    buf = []
    
    
    boll_dev = 2
    current_spread = 0.0
    boll_mid = 0.0
    boll_down = 0.0
    boll_up = 0.0

    
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
        self.last_tick_dict = {key: None for key in vt_symbols}
        self.symbol_to_order_dict = {self.leg1_symbol:[], self.leg2_symbol:[]}

        def on_bar(bar: BarData):
            """"""
            pass
        
        self.ams: dict[str, ArrayManager] = {}
        for vt_symbol in self.vt_symbols:
            self.bgs[vt_symbol] = BarGenerator(on_bar)
            self.ams[vt_symbol] = ArrayManager()
            
        self.pbg = PortfolioBarGenerator(self.on_pbars, 4, self.on_4hour_bars, Interval.HOUR)

    def on_pbars(self, bars: dict[str, BarData]) -> None:
        self.pbg.update_bars(bars)
    
    #TODO should be one day but I don't know why there isn't such a choice...
    def on_4hour_bars(self, bars: dict[str, BarData]) -> None:
        for vt_symbol, bar in bars.items():
            am: ArrayManager = self.ams[vt_symbol]
            am.update_bar(bar)
            if not am.inited:
                return
            
            self.buf = am.close
    
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
                    
    def update_order(self, order: OrderData) -> None:
        pre_order_type,pre_order_status = None,None
        if order.vt_orderid in self.orders:
            pre_order_type,pre_order_status = self.orders[order.vt_orderid].type, self.orders[order.vt_orderid].status
        self.orders[order.vt_orderid] = order

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)
        
        self.write_log_trading(f'pre_order_type {pre_order_type}, pre_order_status {pre_order_status}')
        ##@TODO need multiple rejection counter
        if pre_order_type and pre_order_status and pre_order_type == OrderType.FAK and pre_order_status == Status.SUBMITTING and order.status == Status.CANCELLED and (self.last_tick_dict[order.vt_symbol]):
            self.write_log_trading('update_order')
            last_tick = self.last_tick_dict[order.vt_symbol]
            self.rebalance(order.vt_symbol, last_tick.ask_price_1, last_tick.bid_price_1, 'simple_buy', 'test')
        
    def update_trade(self, trade: TradeData) -> None:
        """成交数据更新"""
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume
            
        if (self.get_pos(trade.vt_symbol) != self.get_target(trade.vt_symbol)) and (self.last_tick_dict[trade.vt_symbol]):
            self.write_log_trading(f'update_trade, self.get_pos(trade.vt_symbol){self.get_pos(trade.vt_symbol)}, self.get_target(trade.vt_symbol){self.get_target(trade.vt_symbol)}')
            last_tick = self.last_tick_dict[trade.vt_symbol]
            self.rebalance(trade.vt_symbol, last_tick.ask_price_1, last_tick.bid_price_1, 'simple_buy', 'test')
            
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
        self.pbg.update_tick(tick)
        self.last_tick_dict[tick.vt_symbol] = tick
        
    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        leg1_bar = bars.get(self.leg1_symbol, None)
        leg2_bar = bars.get(self.leg2_symbol, None)
        if not leg1_bar or not leg2_bar:
            return

        self.set_target(self.leg1_symbol, -self.fixed_size)
        self.set_target(self.leg2_symbol, self.fixed_size)
        # self.rebalance_portfolio_FAK(bars,'simple_buy','test')
        self.rebalance(self.leg1_symbol, leg1_bar.close_price, leg1_bar.close_price, 'simple_buy', 'test')
        self.rebalance(self.leg2_symbol, leg2_bar.close_price, leg2_bar.close_price, 'simple_buy', 'test')
        
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

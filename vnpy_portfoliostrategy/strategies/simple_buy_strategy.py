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
    
    def exe_FAK(self, tick:TickData, order:OrderData = None) -> tuple:
        '''return (buy_price, sell_price) tuple'''
        if order:
            rej_count = order.rejection_count 
        else:
            rej_count = 0
            
        if rej_count >=3:
            raise Exception("FAK seems not able to work")
        
        min_tick:float = self.get_pricetick(tick.vt_symbol)
        bp = tick.ask_price_1 + (rej_count // 2) * min_tick
        sp = tick.bid_price_1 - (rej_count // 2) * min_tick
        return (bp,sp)
        
    def update_order(self, order: OrderData) -> None:
        pre_order_type,pre_order_status = None,None
        if order.vt_orderid in self.orders:
            pre_order_type,pre_order_status = self.orders[order.vt_orderid].type, self.orders[order.vt_orderid].status
        self.orders[order.vt_orderid] = order

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)
        
        if pre_order_type and pre_order_status and pre_order_type == OrderType.FAK and pre_order_status == Status.SUBMITTING and order.status == Status.CANCELLED and (self.last_tick_dict[order.vt_symbol]):
            last_tick = self.last_tick_dict[order.vt_symbol]
            order.rejection_count += 1
            try:
                bp,sp = self.exe_FAK(last_tick, order)
                self.write_log_trading(f'update_order {last_tick.bid_price_1},{last_tick.ask_price_1},{order.rejection_count}')
                self.rebalance(order.vt_symbol, bp, sp, 'simple_buy', 'test')
            except Exception as e:
                print(e)
        
    def update_trade(self, trade: TradeData) -> None:
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume
        
        #TODO this is for partial fill logic
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
            
        self.last_tick_time = tick.datetime
        self.last_tick_dict[tick.vt_symbol] = tick

        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)


        
    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        leg1_bar = bars.get(self.leg1_symbol, None)
        leg2_bar = bars.get(self.leg2_symbol, None)
        if not leg1_bar or not leg2_bar:
            return

        self.set_target(self.leg1_symbol, -self.fixed_size)
        self.set_target(self.leg2_symbol, self.fixed_size)
        if self.last_tick_dict[self.leg1_symbol] and self.last_tick_dict[self.leg2_symbol]:
            self.write_log_trading(f'on_bars {self.last_tick_dict[self.leg1_symbol].bid_price_1},{self.last_tick_dict[self.leg1_symbol].ask_price_1},{leg1_bar.close_price},{self.last_tick_dict[self.leg2_symbol].bid_price_1},{self.last_tick_dict[self.leg2_symbol].ask_price_1},{leg2_bar.close_price}')
            bp,sp = self.exe_FAK(self.last_tick_dict[self.leg1_symbol])
            self.rebalance(self.leg1_symbol,bp,sp, 'simple_buy', 'test')
            bp,sp = self.exe_FAK(self.last_tick_dict[self.leg2_symbol])
            self.rebalance(self.leg2_symbol,bp,sp, 'simple_buy', 'test')
        else:
            self.rebalance(self.leg1_symbol, leg1_bar.close_price, leg1_bar.close_price, 'simple_buy', 'test')
            self.rebalance(self.leg2_symbol, leg2_bar.close_price, leg2_bar.close_price, 'simple_buy', 'test')
        
        # not sure whether necessary
        self.put_event()
        

    def calculate_price(self, vt_symbol: str, direction: Direction, reference: float) -> float:
        return reference

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
from vnpy_portfoliostrategy.helperclass import *

# only to test FAK
class SimpleBuyStrategy(StrategyTemplate):
    
    quantity = '0'
    
    parameters = [
        "quantity"
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        
        self.bool_dict = BoolDict(vt_symbols)
        qty_list = self.quantity.split(',')
        if len(qty_list) != len(self.vt_symbols):
            raise Exception('wrong length of quantity and vt_symbols')
        for ix,symb in enumerate(self.vt_symbols):
            self.set_target(symb, int(qty_list[ix]))
            self.write_log(f"symb {symb} curpos = {self.get_pos(symb)}")
            self.write_log(f"symb {symb} tarpos = {self.get_target(symb)}")
    
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
        
    def update_order(self, order: OrderData) -> None:
        """委托数据更新"""
        symb = order.vt_symbol
        self.orders[order.vt_orderid] = order
        self.write_log(f"order id, datetime, type, status {order.vt_orderid}, {order.datetime},{order.type}, {order.status}")

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)
            self.symbol_status[symb].is_active = False
        
        if (order.type == OrderType.FAK or order.type == OrderType.FOK):
            if order.status == Status.REJECTED:
                self.symbol_status[symb].rej_counts += 1
            if order.status == Status.CANCELLED:
                self.symbol_status[symb].can_counts += 1
            
    def on_tick(self, tick: TickData) -> None:
        # if not self.trading or not super().on_tick(tick):
        #     return
        
        if self.bool_dict.all_true():
            self.strategy_engine.stop_strategy(self.strategy_name,
                                               f"All have rebalanced. Stop the strategy {self.strategy_name} now")
            return
        
        if (self.get_target(tick.vt_symbol) != self.get_pos(tick.vt_symbol)):
            if (not self.symbol_status[tick.vt_symbol].is_active and not self.symbol_status[tick.vt_symbol].is_stop()):
                bp,ap = self.get_retry_price(tick)
                self.rebalance(tick.vt_symbol, bp, ap, net=False, strategy='simple_buy',intention='rebalance')
        else:
            self.bool_dict.set(tick.vt_symbol, True)


from datetime import datetime

import numpy as np
import pandas as pd
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

class Strategy1(StrategyTemplate):
    """配对交易策略"""
    
    tick_add = 0
    fixed_size = 1
    buf=None
    bar_counter = 0
    bar_freq = 240 #@TODO need a trading hour table
    sample_n = 3
    
    
    boll_mid = 0.0
    boll_down = 0.0
    boll_up = 0.0
    
    boll_window =  10
    boll_dev = 2

    parameters = [
        "boll_window",
        "boll_dev",
    ]

    
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
        self.samp_am = {self.leg1_symbol:[], self.leg2_symbol:[]}

        def on_bar(bar: BarData):
            """"""
            pass
        
        self.amss: dict[int, dict] = {}
        ams: dict[str, ArrayManager] = {}
        for vt_symbol in self.vt_symbols:
            self.bgs[vt_symbol] = BarGenerator(on_bar)
            for i in range(self.sample_n):
                ams[vt_symbol] = ArrayManager(size=self.boll_window) 
                self.amss[i] = ams
        

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.load_bars(30) # this is number of natural days in real life and number of trading days in backtesting, need to be large
        self.put_event()  # this is to update GUI

    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")
        self.write_log(f"trading instruments - {self.vt_symbols[0]},{self.vt_symbols[1]}")
        self.write_log(f"ams close {self.amss[0][self.vt_symbols[0]].close}")
        self.put_event()

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
        
        if pre_order_type and pre_order_status and pre_order_type == OrderType.FAK and pre_order_status == Status.SUBMITTING and order.status == Status.CANCELLED and (self.last_tick_dict[order.vt_symbol]):
            last_tick = self.last_tick_dict[order.vt_symbol]
            order.rejection_count += 1
            try:
                bp,sp = self.exe_FAK(last_tick, order)
                self.rebalance(order.vt_symbol, bp, sp, 'strategy1', 'boll')
            except Exception as e:
                print(e)
                
    def update_trade(self, trade: TradeData) -> None:
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume
        
        #TODO this is for partial fill logic
        if (self.get_pos(trade.vt_symbol) != self.get_target(trade.vt_symbol)) and (self.last_tick_dict[trade.vt_symbol]):
            last_tick = self.last_tick_dict[trade.vt_symbol]
            self.rebalance(trade.vt_symbol,last_tick.ask_price_1,last_tick.bid_price_1, 'strategy1', 'boll')
            
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


    # Only for 1 minute bar
    def on_bars(self, bars: dict[str, BarData]) -> None:
        self.put_event()
        leg1_bar = bars.get(self.leg1_symbol, None)
        leg2_bar = bars.get(self.leg2_symbol, None)
        if not leg1_bar or not leg2_bar:
            return
        
        ## TODO what if the market data disconnect
        if self.buf is not None:
            current_spread = leg1_bar.close_price- leg2_bar.close_price
            self.cal_target_pos(current_spread, bars)
            self.write_log_trading(f'self.boll_mid {self.boll_mid}, self.boll_up {self.boll_up}, self.boll_down {self.boll_down}, current spread {current_spread}')

        # Ideally should be the last minute
        for i in range(self.sample_n):
            if leg1_bar.datetime.hour == 14 and leg1_bar.datetime.minute == 59-i:
                self.on_win_bars(bars, self.amss[i])
            

    def on_win_bars(self, bars: dict[str, BarData], am: dict[str, ArrayManager]) -> None:
        am1 = am[self.leg1_symbol]
        am2 = am[self.leg2_symbol]
        am1.update_bar(bars[self.leg1_symbol])
        am2.update_bar(bars[self.leg2_symbol])
        if (not am1.inited) or (not am2.inited):
            return
        
        self.buf = am1.close - am2.close
        
        std = self.buf.std()
        self.boll_mid = self.buf.mean()
        self.boll_up = self.boll_mid + self.boll_dev * std
        self.boll_down = self.boll_mid - self.boll_dev * std
                
    def need_to_rebalance(self, tar1, tar2, bars: dict[str, BarData]) -> None: 
        self.write_log_trading(f"Need to rebalance {tar1}, {tar2}, {self.get_pos(self.leg1_symbol)}, {self.get_pos(self.leg2_symbol)}")
        if self.get_pos(self.leg1_symbol)!=tar1:
            self.set_target(self.leg1_symbol, tar1)
            if self.last_tick_dict[self.leg1_symbol]:
                bp,sp = self.exe_FAK(self.last_tick_dict[self.leg1_symbol])
                self.rebalance(self.leg1_symbol,bp,sp, 'strategy1', 'boll')
            else:
                bar = bars.get(self.leg1_symbol, None)
                self.rebalance(self.leg1_symbol, bar.close_price, bar.close_price, 'strategy1', 'boll')
            
        if self.get_pos(self.leg2_symbol)!=tar2:
            self.set_target(self.leg2_symbol, tar2)
            if self.last_tick_dict[self.leg2_symbol]:
                bp,sp = self.exe_FAK(self.last_tick_dict[self.leg2_symbol])
                self.rebalance(self.leg2_symbol,bp,sp, 'strategy1', 'boll')
            else:
                bar = bars.get(self.leg2_symbol, None)
                self.rebalance(self.leg2_symbol, bar.close_price, bar.close_price, 'strategy1', 'boll')
    
    # TODO ideally should be in tick, using bid-ask instead of close price
    def cal_target_pos(self, current_spread:float, bars: dict[str, BarData]) -> None:
        leg1_pos = self.get_pos(self.leg1_symbol)
        leg2_pos = self.get_pos(self.leg2_symbol)
        if leg1_pos == 0 and leg2_pos == 0:
            if current_spread >= self.boll_up:
                tar1,tar2 = -self.fixed_size,self.fixed_size
                self.need_to_rebalance(tar1, tar2, bars)
            elif current_spread <= self.boll_down:
                tar1,tar2 = self.fixed_size, -self.fixed_size
                self.need_to_rebalance(tar1, tar2, bars)
        elif leg1_pos > 0 and leg2_pos < 0:
            if current_spread >= self.boll_mid:
                tar1, tar2 = 0,0
                self.need_to_rebalance(tar1, tar2, bars)
        elif leg1_pos < 0 and leg2_pos > 0:
            if current_spread <= self.boll_mid:
                tar1, tar2 = 0,0
                self.need_to_rebalance(tar1, tar2, bars)
        # TODO May need to sync database if otherwise
                
        
    def calculate_price(self, vt_symbol: str, direction: Direction, reference: float) -> float:
        return reference

    
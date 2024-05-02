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
class Strategy1(StrategyTemplate):
    """配对交易策略"""
    
    tick_add = 0
    fixed_size = 1
    buf=None
    
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
        self.samp_am = {self.leg1_symbol:[], self.leg2_symbol:[]}

        def on_bar(bar: BarData):
            """"""
            pass
        
        self.ams: dict[str, ArrayManager] = {}
        for vt_symbol in self.vt_symbols:
            self.bgs[vt_symbol] = BarGenerator(on_bar)
            self.ams[vt_symbol] = ArrayManager(size=30)  #TODO change this

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.load_bars(10) #TODO change this

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
        
        ##@TODO need multiple rejection counter
        if pre_order_type and pre_order_status and pre_order_type == OrderType.FAK and pre_order_status == Status.SUBMITTING and order.status == Status.CANCELLED:
            self.rebalance(order.vt_symbol, self.last_tick_dict[order.vt_symbol])
        
    def update_trade(self, trade: TradeData) -> None:
        """成交数据更新"""
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume
            
        if (self.get_pos(trade.vt_symbol) != self.get_target(trade.vt_symbol)) and (self.last_tick_dict[trade.vt_symbol]):
            self.rebalance(trade.vt_symbol,self.last_tick_dict[trade.vt_symbol])
            
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
        self.last_tick_dict[tick.vt_symbol] = tick

    def on_bars(self, bars: dict[str, BarData]) -> None:
        def check_timestamp_at_freq(dt:datetime) -> bool:
            if dt.hour == 14 and dt.minute == 59:
                return True
            else:
                return False
                
        leg1_bar = bars.get(self.leg1_symbol, None)
        leg2_bar = bars.get(self.leg2_symbol, None)
        if not leg1_bar or not leg2_bar:
            return
        
        for vt_symbol, bar in bars.items():
            am: ArrayManager = self.ams[vt_symbol]
            am.update_bar(bar)   # TODO probably don't need this at all
            # if not am.inited:   # TODO start when X bars has passed, need this
            #     return
            
            if not self.buf:
                for c in am.close:
                    if check_timestamp_at_freq(datetime.fromtimestamp(c)):  # TODO bad practive
                        self.samp_am[vt_symbol].append(c)
            else:
                if check_timestamp_at_freq(bar.datetime):
                     self.samp_am[vt_symbol][:-1] = self.samp_am[vt_symbol][1:]
                     self.samp_am[vt_symbol][-1] = bar.close_price
                      
        if (not self.buf) and self.samp_am[self.leg1_symbol] and self.samp_am[self.leg2_symbol]:
            self.buf = np.array(self.samp_am[self.leg1_symbol]) - np.array(self.samp_am[self.leg2_symbol])  #TODO double check this logic
        elif self.buf and self.samp_am[self.leg1_symbol] and self.samp_am[self.leg2_symbol]: 
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = self.samp_am[self.leg1_symbol][-1] - self.samp_am[self.leg2_symbol][-1]
                
        ## TODO what if the market data disconnect
        if self.buf:
            std = self.buf.std()
            self.boll_mid = self.buf.mean()
            self.boll_up = self.boll_mid + self.boll_dev * std
            self.boll_down = self.boll_mid - self.boll_dev * std

            current_spread = leg1_bar.close_price- leg2_bar.close_price
            self.cal_target_pos(current_spread, bars)
            # not sure whether necessary
            self.put_event()
    
    # TODO ideally should be in tick
    def cal_target_pos(self, current_spread:float, bars) -> None:
        leg1_pos = self.get_pos(self.leg1_symbol)
        if not leg1_pos:
            if current_spread >= self.boll_up:
                self.set_target(self.leg1_symbol, -self.fixed_size)
                self.set_target(self.leg2_symbol, self.fixed_size)
            elif self.current_spread <= self.boll_down:
                self.set_target(self.leg1_symbol, self.fixed_size)
                self.set_target(self.leg2_symbol, -self.fixed_size)
        elif leg1_pos > 0:
            if current_spread >= self.boll_mid:
                self.set_target(self.leg1_symbol, 0)
                self.set_target(self.leg2_symbol, 0)
        else:
            if current_spread <= self.boll_mid:
                self.set_target(self.leg1_symbol, 0)
                self.set_target(self.leg2_symbol, 0)
        self.rebalance_portfolio_FAK(bars)
        
    def calculate_price(self, vt_symbol: str, direction: Direction, reference: float) -> float:
        """计算调仓委托价格（支持按需重载实现）"""
        pricetick: float = self.get_pricetick(vt_symbol)

        if direction == Direction.LONG:
            price: float = reference + self.tick_add * pricetick
        else:
            price: float = reference - self.tick_add * pricetick

        return price

    def rebalance(self, vt_symbol: str, tick:TickData) -> None:
        """基于目标执行调仓交易"""
        ##@TODO need partial fill logic
        for oid in self.symbol_to_order_dict[vt_symbol]:
            if oid in self.orders and self.orders[oid].status == Status.NOTTRADED:
                self.cancel_order(oid)

        result_list: list = []
        target: int = self.get_target(vt_symbol)
        pos: int = self.get_pos(vt_symbol)
        diff: int = target - pos

        # 多头
        if diff > 0:
            # 计算多头委托价
            order_price: float = self.calculate_price(
                vt_symbol,
                Direction.LONG,
                tick.ask_price_1
            )

            # 计算买平和买开数量
            cover_volume: int = 0
            buy_volume: int = 0

            if pos < 0:
                cover_volume = min(diff, abs(pos))
                buy_volume = diff - cover_volume
            else:
                buy_volume = diff

            # 发出对应委托
            if cover_volume:
                result_list = self.cover(vt_symbol, order_price, cover_volume, isFAK=True)

            if buy_volume:
                result_list = self.buy(vt_symbol, order_price, buy_volume, isFAK=True)
        # 空头
        elif diff < 0:
            # 计算空头委托价
            order_price: float = self.calculate_price(
                vt_symbol,
                Direction.SHORT,
                tick.bid_price_1
            )

            # 计算卖平和卖开数量
            sell_volume: int = 0
            short_volume: int = 0

            if pos > 0:
                sell_volume = min(abs(diff), pos)
                short_volume = abs(diff) - sell_volume
            else:
                short_volume = abs(diff)

            # 发出对应委托
            if sell_volume:
                result_list = self.sell(vt_symbol, order_price, sell_volume, isFAK=True)

            if short_volume:
                result_list = self.short(vt_symbol, order_price, short_volume, isFAK=True)
        
        self.symbol_to_order_dict[vt_symbol].append(result_list[-1])

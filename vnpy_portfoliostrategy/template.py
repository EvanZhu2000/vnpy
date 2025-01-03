from abc import ABC
from copy import copy
from typing import TYPE_CHECKING, Optional,DefaultDict, List
from collections import defaultdict
from datetime import datetime, timedelta, time
from vnpy.trader.constant import Interval, Direction, Offset, Status
from vnpy.trader.object import BarData, TickData, OrderData, TradeData, OrderType
from vnpy.trader.utility import virtual
from vnpy_portfoliostrategy.base import EngineType
from vnpy_portfoliostrategy.helperclass import *
from collections import defaultdict
import re
import pytz
import pandas as pd

if TYPE_CHECKING:
    from vnpy_portfoliostrategy.engine import StrategyEngine

class StrategyTemplate(ABC):
    """组合策略模板"""

    author: str = ""
    parameters: list = []
    variables: list = []

    def __init__(
        self,
        strategy_engine: "StrategyEngine",
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """构造函数"""
        self.strategy_engine: "StrategyEngine" = strategy_engine
        self.strategy_name: str = strategy_name
        self.starting_time: datetime = None
        self.vt_symbols: list[str] = vt_symbols
        self.trading_hours: dict[str, str] = None  # symb: trading hours
        self.rebal_tracker: BoolDict = None
        self.settlement_dates_str = None

        # Hard code stuff
        self.time_since_last_tick = timedelta(seconds=60) 
        self.time_since_starting = timedelta(minutes=10) 
        
        # 状态控制变量
        self.inited: bool = False
        self.trading: bool = False

        # 持仓数据字典
        self.pos_data: dict[str, int] = defaultdict(int)        # 实际持仓
        self.target_data: dict[str, int] = defaultdict(int)     # 目标持仓

        # 委托缓存容器
        self.orders: dict[str, OrderData] = {}
        self.trades: DefaultDict[datetime, list[TradeData]] = defaultdict(list)
        self.active_orderids: set[str] = set()
        self.symbol_status: dict[str, SymbolStatus] = {}
        for v in self.vt_symbols:
            self.symbol_status[v] = SymbolStatus()

        # 复制变量名列表，插入默认变量内容
        self.variables: list = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos_data")
        self.variables.insert(3, "target_data")

        # 设置策略参数
        self.update_setting(setting)
    
    def update_setting(self, setting: dict) -> None:
        """设置策略参数"""
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls) -> dict:
        """查取策略默认参数"""
        class_parameters: dict = {}
        for name in cls.parameters:
            class_parameters[name] = getattr(cls, name)
        return class_parameters

    def get_parameters(self) -> dict:
        """查询策略参数"""
        strategy_parameters: dict = {}
        for name in self.parameters:
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self) -> dict:
        """查询策略变量"""
        strategy_variables: dict = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self) -> dict:
        """查询策略状态数据"""
        strategy_data: dict = {
            "strategy_name": self.strategy_name,
            "vt_symbols": self.vt_symbols,
            "class_name": self.__class__.__name__,
            "author": self.author,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return strategy_data

    @virtual
    def on_init(self) -> None:
        """策略初始化回调"""
        pass

    @virtual
    def on_start(self) -> None:
        """策略启动回调"""
        if self.strategy_engine.engine_type == EngineType.LIVE:
            self.starting_time = datetime.now(pytz.timezone('Asia/Shanghai'))
        elif self.strategy_engine.engine_type == EngineType.BACKTESTING:
            self.starting_time = self.strategy_engine.starting_time  # specific for backtesting
            
        self.write_log(f"Strategy {self.strategy_name} started at {self.starting_time}")
        pass

    @virtual
    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log(f"FINAL pos_data {self.nonzero_dict(self.pos_data)}")
        self.write_log(f"FINAL target_data {self.nonzero_dict(self.target_data)}")
        self.write_log(f"FINAL diff {pd.concat([pd.Series(self.pos_data),pd.Series(self.target_data)],axis=1,keys=['pos','tar']).query('pos!=tar').to_dict()}")
        if self.strategy_engine.engine_type == EngineType.LIVE:
            rows = [
                (date, tr.datetime, tr.vt_symbol, tr.vt_orderid, tr.direction, tr.offset, tr.price, tr.volume)
                for date, trades in self.trades.items()
                for tr in trades
            ]
            trade_records = pd.DataFrame(rows, columns=['signal_datetime', 'datetime','vt_symbol', 'vt_orderid','direction','offset','price', 'volume'])
            trade_records[['direction','offset']] = trade_records[['direction','offset']].astype(str)
            
            self.strategy_engine.dbservice.init_connection()
            self.strategy_engine.dbservice.update_pos(self.pos_data)  # only to update pos_data
            self.strategy_engine.dbservice.insert_rq(trade_records, 'trade_records', ignore=True)
            self.strategy_engine.dbservice.close()

    @virtual
    # return False when the tick has some issues
    def on_tick(self, tick: TickData) -> bool:
        """行情推送回调"""
        if not self.check_valid_tick(tick):
            return False
        
        # whether the subscription is successful
        if self.symbol_status[tick.vt_symbol].last_tick is None:
            self.write_log(f'first tick for {tick.vt_symbol} is {tick}')
            self.symbol_status[tick.vt_symbol].last_tick = tick
        # feed check
        elif tick.datetime - self.symbol_status[tick.vt_symbol].last_tick.datetime < timedelta(minutes=0):
            return False
        elif tick.datetime - self.symbol_status[tick.vt_symbol].last_tick.datetime > self.time_since_last_tick \
            and (self.symbol_status[tick.vt_symbol].alarm_time_since_last_tick is None or tick.datetime - self.symbol_status[tick.vt_symbol].alarm_time_since_last_tick > self.time_since_last_tick):
            self.write_log_level1(f'Too long since last tick for {tick.vt_symbol}, breaching {self.time_since_last_tick}, last tick time: {self.symbol_status[tick.vt_symbol].last_tick.datetime}, current time: {tick.datetime}')
            self.symbol_status[tick.vt_symbol].alarm_time_since_last_tick = tick.datetime
            
        return True


    @virtual
    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        pass

    def update_trade(self, trade: TradeData) -> None:
        """成交数据更新"""
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume
            
        if self.rebal_tracker is not None and self.rebal_tracker.target_time_dict is not None:
            self.trades[self.rebal_tracker.target_time_dict[trade.vt_symbol]].append(trade)
            

    def update_order(self, order: OrderData) -> None:
        """委托数据更新"""
        symb = order.vt_symbol
        self.orders[order.vt_orderid] = order

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)
            self.symbol_status[symb].is_active = False
        
        if (order.type == OrderType.FAK or order.type == OrderType.FOK):
            if order.status == Status.REJECTED:
                self.symbol_status[symb].rej_counts += 1
            if order.status == Status.CANCELLED:
                self.symbol_status[symb].can_counts += 1
            
        # self.strategy_engine.dbservice.update_order_status(order.vt_orderid, order.status)

    def buy(self, vt_symbol: str, price: float, volume: float, lock: bool = False, net: bool = False, isFAK: bool = False,strategy:str = None,intention:str = None,pos=None,tar=None) -> list[str]:
        """买入开仓"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume, lock, net, isFAK,strategy,intention,pos,tar)

    def sell(self, vt_symbol: str, price: float, volume: float, lock: bool = False, net: bool = False, isFAK: bool = False,strategy:str = None,intention:str = None,pos=None,tar=None) -> list[str]:
        """卖出平仓"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume, lock, net, isFAK,strategy,intention,pos,tar)

    def short(self, vt_symbol: str, price: float, volume: float, lock: bool = False, net: bool = False, isFAK: bool = False,strategy:str = None,intention:str = None,pos=None,tar=None) -> list[str]:
        """卖出开仓"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume, lock, net, isFAK,strategy,intention,pos,tar)

    def cover(self, vt_symbol: str, price: float, volume: float, lock: bool = False, net: bool = False, isFAK: bool = False,strategy:str = None,intention:str = None,pos=None,tar=None) -> list[str]:
        """买入平仓"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume, lock, net, isFAK,strategy,intention,pos,tar)

    def send_order(
        self,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool = False,
        net: bool = False,
        isFAK: bool = False,
        strategy:str = None,
        intention:str = None,
        pos=None,
        tar=None
    ) -> list[str]:
        """发送委托"""
        if self.trading:
            try:
                # # order number check
                # if len(self.symbol_status[vt_symbol].order_list) > 20:
                #     self.strategy_engine.write_exception(f"Too much orders for {vt_symbol}, stop sending additional orders!")
                    
                if isFAK:
                    vt_orderids: list = self.strategy_engine.send_order_FAK(
                        self, vt_symbol, direction, offset, price, volume, lock, net
                    )
                else:
                    vt_orderids: list = self.strategy_engine.send_order(
                        self, vt_symbol, direction, offset, price, volume, lock, net
                    )

                for vt_orderid in vt_orderids:
                    self.active_orderids.add(vt_orderid)
                    self.symbol_status[vt_symbol].is_active = True
                    self.symbol_status[vt_symbol].order_list.append(vt_orderid)

                return vt_orderids
            except Exception as e:
                self.strategy_engine.write_log(f"Exception when sending order - {e}")
                return []
        else:
            return []

    def cancel_order(self, vt_orderid: str) -> None:
        """撤销委托"""
        try:
            if self.trading:
                self.strategy_engine.cancel_order(self, vt_orderid)
        
        except Exception as e:
            self.strategy_engine.write_log(f"Exception when sending order - {e}")

    def cancel_all(self) -> None:
        """全撤活动委托"""
        for vt_orderid in list(self.active_orderids):
            self.cancel_order(vt_orderid)
        for k,_ in self.symbol_status.items():
            self.symbol_status[k].is_active = False

    def get_pos(self, vt_symbol: str) -> int:
        """查询当前持仓"""
        return self.pos_data.get(vt_symbol, 0)

    def get_target(self, vt_symbol: str) -> int:
        """查询目标仓位"""
        return self.target_data[vt_symbol]
    
    def set_pos(self, vt_symbol: str, pos: int) -> None:
        """when we restart the strategy we want to sync db for positions"""
        self.pos_data[vt_symbol] = pos

    def set_target(self, vt_symbol: str, target: int) -> None:
        """设置目标仓位"""
        self.target_data[vt_symbol] = target

    def get_retry_price(self, tick:TickData) -> tuple:
        '''return (buy_price, sell_price) tuple'''
        vt_symbol = tick.vt_symbol
        rej_count = self.symbol_status[vt_symbol].rej_counts
        can_count = self.symbol_status[vt_symbol].can_counts
        # Something wrong with the system
        if rej_count >=3:
            self.on_stop()
            self.strategy_engine.stop_strategy(self.strategy_name, 
                                               f"reject counts >=3 for {vt_symbol}",
                                               f"{self.strategy_name}_fail_{self.strategy_engine.main_engine.env}")
        # the book moved so fast
        if can_count >=5:
            self.symbol_status[vt_symbol].stop_FAK_cancel = True
            self.rebal_tracker.true_count += 1
            self.write_log_level1('can_count breaching limit, {vt_symbol} stop_FAK_cancel')
        
        min_tick:float = self.get_pricetick(tick.vt_symbol)
        tmp = min(int(can_count // 2), 2)
        bp = tick.ask_price_1 + tmp * min_tick
        sp = tick.bid_price_1 - tmp * min_tick
        return (bp,sp)
    
    def rebalance(self, vt_symbol: str, buy_price:float, sell_price:float, net:bool=False, strategy:str=None, intention:str=None) -> None:
        """基于目标执行调仓交易"""
        if self.symbol_status[vt_symbol].is_active:
            pass

        target: int = self.get_target(vt_symbol)
        pos: int = self.get_pos(vt_symbol)
        diff: int = target - pos

        # 多头
        if diff > 0:
            order_price: float = self.calculate_price(
                vt_symbol,
                Direction.LONG,
                buy_price
            )

            cover_volume: int = 0
            buy_volume: int = 0

            if pos < 0:
                cover_volume = min(diff, abs(pos))
                buy_volume = diff - cover_volume
            else:
                buy_volume = diff

            if cover_volume:
                self.cover(vt_symbol, order_price, cover_volume, net=net, isFAK=True, strategy=strategy,intention=intention,pos=pos,tar=target)

            if buy_volume:
                self.buy(vt_symbol, order_price, buy_volume, net=net, isFAK=True, strategy=strategy,intention=intention,pos=pos,tar=target)
        # 空头
        elif diff < 0:
            order_price: float = self.calculate_price(
                vt_symbol,
                Direction.SHORT,
                sell_price
            )

            sell_volume: int = 0
            short_volume: int = 0

            if pos > 0:
                sell_volume = min(abs(diff), pos)
                short_volume = abs(diff) - sell_volume
            else:
                short_volume = abs(diff)

            if sell_volume:
                self.sell(vt_symbol, order_price, sell_volume, net=net, isFAK=True, strategy=strategy,intention=intention,pos=pos,tar=target)

            if short_volume:
                self.short(vt_symbol, order_price, short_volume, net=net, isFAK=True, strategy=strategy,intention=intention,pos=pos,tar=target)


    def rebalance_portfolio(self, bars: dict[str, BarData]) -> None:
        """基于目标执行调仓交易"""
        self.cancel_all()

        # 只发出当前K线切片有行情的合约的委托
        for vt_symbol, bar in bars.items():
            # 计算仓差
            target: int = self.get_target(vt_symbol)
            pos: int = self.get_pos(vt_symbol)
            diff: int = target - pos

            # 多头
            if diff > 0:
                # 计算多头委托价
                order_price: float = self.calculate_price(
                    vt_symbol,
                    Direction.LONG,
                    bar.close_price
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
                    self.cover(vt_symbol, order_price, cover_volume)

                if buy_volume:
                    self.buy(vt_symbol, order_price, buy_volume)
            # 空头
            elif diff < 0:
                # 计算空头委托价
                order_price: float = self.calculate_price(
                    vt_symbol,
                    Direction.SHORT,
                    bar.close_price
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
                    self.sell(vt_symbol, order_price, sell_volume)

                if short_volume:
                    self.short(vt_symbol, order_price, short_volume)

    @virtual
    def calculate_price(
        self,
        vt_symbol: str,
        direction: Direction,
        reference: float
    ) -> float:
        """计算调仓委托价格（支持按需重载实现）"""
        return reference

    def get_order(self, vt_orderid: str) -> Optional[OrderData]:
        """查询委托数据"""
        return self.orders.get(vt_orderid, None)

    def get_all_active_orderids(self) -> list[OrderData]:
        """获取全部活动状态的委托号"""
        return list(self.active_orderids)

    def write_log(self, msg: str) -> None:
        """记录日志"""
        self.strategy_engine.write_log(msg, self)
    
    # write log along with level 1
    def write_log_level1(self, msg: str) -> None:
        self.strategy_engine.write_log(msg, self)
        # TODO level 1 warning
        
    def write_log_trading(self, msg: str) -> None:
        "Recording debugging logs in trading period only"
        if self.trading:
            self.write_log(msg=msg)

    def get_engine_type(self) -> EngineType:
        """查询引擎类型"""
        return self.strategy_engine.get_engine_type()

    def get_pricetick(self, vt_symbol: str) -> float:
        """查询合约最小价格跳动"""
        return self.strategy_engine.get_pricetick(self, vt_symbol)

    def get_size(self, vt_symbol: str) -> int:
        """查询合约乘数"""
        return self.strategy_engine.get_size(self, vt_symbol)

    def load_bars(self, days: int, interval: Interval = Interval.MINUTE) -> None:
        """加载历史K线数据来执行初始化"""
        self.strategy_engine.load_bars(self, days, interval)

    def put_event(self) -> None:
        """推送策略数据更新事件"""
        if self.inited:
            self.strategy_engine.put_strategy_event(self)

    def send_email(self, msg: str) -> None:
        """发送邮件信息"""
        # Below is the original vnpy way of sending order
        if self.inited:
            self.strategy_engine.send_email(msg, self)

    def sync_data(self) -> None:
        """同步策略状态数据到文件"""
        if self.trading:
            self.strategy_engine.sync_strategy_data(self)
    
    # return nonzero version of a dict (usually refers to dict of positions)      
    def nonzero_dict(self, d) -> dict:
        non_zero_items = {k: v for k, v in d.items() if v != 0}
        sorted_non_zero_items = dict(sorted(non_zero_items.items()))
        return sorted_non_zero_items
    
    def check_valid_tick(self, tick) -> bool:
        if self.trading_hours is None:
            self.strategy_engine.stop_strategy(self.strategy_name, 
                                               f"No trading hours. Stop the strategy {self.strategy_name} now",
                                               f"{self.strategy_name}_fail_{self.strategy_engine.main_engine.env}")
            return False
        
        # Check whether the tick is in continuous trading hours
        effective_symbol = re.sub(r'\d+', '', tick.vt_symbol)
        if effective_symbol not in self.trading_hours.keys():
            self.strategy_engine.stop_strategy(self.strategy_name, 
                                               f"No trading hours provided for {tick.vt_symbol}, effective symbol is {effective_symbol}. Stop the strategy {self.strategy_name} now",
                                               f"{self.strategy_name}_fail_{self.strategy_engine.main_engine.env}")
            return False
        else:
            continuous_trading_intervals = self.trading_hours[effective_symbol]
            if not self.is_time_in_intervals(tick.datetime.time(), continuous_trading_intervals):
                # Then this tick is not a continuous trading tick
                return False
        
        # Check whether the tick is valid   
        if not tick.last_price:
            return False
        return True

    # Check whether input_time is in intervals of time, along with minus seconds customization for start/end time
    def is_time_in_intervals(self, input_time, intervals, start_time_minus_seconds=60, end_time_minus_seconds=10) -> bool:
        '''
        start_time_minus_seconds=60 : ricequant unique trading hours
        end_time_minus_seconds=10 : last 10 seconds let's not trade
        '''
        for interval in intervals.split(','):
            start_str, end_str = interval.split('-')
            
            start_time = datetime.strptime(start_str, '%H:%M').time()
            end_time = datetime.strptime(end_str, '%H:%M').time()
            
            adjusted_start_time = (datetime.combine(datetime.today(), start_time) - timedelta(seconds=start_time_minus_seconds)).time()
            adjusted_end_time = (datetime.combine(datetime.today(), end_time) - timedelta(seconds=end_time_minus_seconds)).time()
            
            # Handle overnight trading periods
            if adjusted_start_time > adjusted_end_time:
                # For overnight trading (e.g. 21:00-03:00), check if time is after start or before end
                if adjusted_start_time <= input_time or input_time <= adjusted_end_time:
                    return True
            else:
                # Normal trading period within same day
                if adjusted_start_time <= input_time <= adjusted_end_time:
                    return True
                
        return False
    
    def get_open_time(self, intervals, start_time_minus_seconds=60) -> datetime:
        try:
            dates, zones = self.settlement_dates_str.split(':')
            date_1, date_2 = dates.split(',')
            
            start_time_str = intervals.split(',')[0].split('-')[0]
            start_time = datetime.strptime(start_time_str, '%H:%M').time()
            adjusted_start_time = (datetime.combine(datetime.today(), start_time) - timedelta(seconds=start_time_minus_seconds)).time()
            
            date_to_use = date_2 if adjusted_start_time <= time(15,30) else date_1
            result = datetime.combine(datetime.strptime(date_to_use, '%Y-%m-%d'), adjusted_start_time)
            result = pytz.timezone(zones).localize(result)
            return result
        except Exception as e:
            self.strategy_engine.write_exception(f'cannot get open time - exception {e}')
            

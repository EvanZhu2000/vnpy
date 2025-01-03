import importlib
import glob
import traceback
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Type, Callable, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from vnpy_self.alert_sender import *

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.object import (
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest,
    LogData,
    TickData,
    OrderData,
    TradeData,
    BarData,
    ContractData
)
from vnpy.trader.event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE
)
from vnpy.trader.constant import (
    Direction,
    OrderType,
    Interval,
    Exchange,
    Offset
)
from vnpy.trader.utility import load_json, save_json, extract_vt_symbol, round_to
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed
from vnpy.trader.database import BaseDatabase, get_database, DB_TZ
from .mysqlservice import MysqlService
from .base import (
    APP_NAME,
    EVENT_PORTFOLIO_LOG,
    EVENT_PORTFOLIO_STRATEGY,
    EngineType
)
from .template import StrategyTemplate
import pandas as pd

class StrategyEngine(BaseEngine):
    """组合策略引擎"""

    engine_type: EngineType = EngineType.LIVE
    main_engine: MainEngine
    setting_filename: str = "portfolio_strategy_setting.json"
    data_filename: str = "portfolio_strategy_data.json"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.strategy_data: dict[str, dict] = {}
        self.main_engine = main_engine
        self.classes: dict[str, Type[StrategyTemplate]] = {}
        self.strategies: dict[str, StrategyTemplate] = {}

        self.symbol_strategy_map: dict[str, list[StrategyTemplate]] = defaultdict(list)
        self.orderid_strategy_map: dict[str, StrategyTemplate] = {}

        self.init_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        self.vt_tradeids: set[str] = set()

        # 数据库和数据服务
        self.database: BaseDatabase = get_database()
        self.datafeed: BaseDatafeed = get_datafeed()
        self.dbservice = MysqlService()

    def init_engine(self) -> None:
        """初始化引擎"""
        self.init_datafeed()
        self.load_strategy_class()
        self.load_strategy_setting()
        # self.load_strategy_data()
        self.register_event()
        self.write_log("组合策略引擎初始化成功")

    def close(self) -> None:
        """关闭"""
        self.stop_all_strategies()

    def register_event(self) -> None:
        """注册事件引擎"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)

    def init_datafeed(self) -> None:
        """初始化数据服务"""
        result: bool = self.datafeed.init(self.write_log)
        if result:
            self.write_log("数据服务初始化成功")

    def query_bar_from_datafeed(
        self, symbol: str, exchange: Exchange, interval: Interval, start: datetime, end: datetime
    ) -> list[BarData]:
        """通过数据服务获取历史数据"""
        req: HistoryRequest = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )
        data: list[BarData] = self.datafeed.query_bar_history(req, self.write_log)
        return data

    def process_tick_event(self, event: Event) -> None:
        """行情数据推送"""
        tick: TickData = event.data

        strategies: list = self.symbol_strategy_map[tick.vt_symbol]
        if not strategies:
            return

        for strategy in strategies:
            if strategy.inited:
                self.call_strategy_func(strategy, strategy.on_tick, tick)
                
        # TODO need a separate rebalancer

    def process_order_event(self, event: Event) -> None:
        """委托数据推送"""
        order: OrderData = event.data

        strategy: Optional[StrategyTemplate] = self.orderid_strategy_map.get(order.vt_orderid, None)
        if not strategy:
            return

        self.call_strategy_func(strategy, strategy.update_order, order)

    def process_trade_event(self, event: Event) -> None:
        """成交数据推送"""
        trade: TradeData = event.data

        # 过滤重复的成交推送
        if trade.vt_tradeid in self.vt_tradeids:
            return
        self.vt_tradeids.add(trade.vt_tradeid)

        # 推送给策略
        strategy: Optional[StrategyTemplate] = self.orderid_strategy_map.get(trade.vt_orderid, None)
        if not strategy:
            return

        self.call_strategy_func(strategy, strategy.update_trade, trade)

    def send_order_FAK(
        self,
        strategy: StrategyTemplate,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool,
        net: bool,
    ) -> list:
        """发送委托"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        if not contract:
            self.write_log(f"委托失败，找不到合约：{vt_symbol}", strategy)
            return ""

        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_volume)

        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=offset,
            type=OrderType.FAK,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{strategy.strategy_name}"
        )

        # NOTE:上期所在平仓的时候需要分别发出平今和平昨指令，而其他交易所平仓指令会自动转换（优先平今）。这个发送是指发给交易所。策略里下单会通过vnpy.trader.converter来自动转换。上期所大概原理是当要求手数少于昨仓，全部平昨。当要求手数多于昨仓，先平昨，剩下的平今
        req_list: list[OrderRequest] = self.main_engine.convert_order_request(
            original_req,
            contract.gateway_name,
            lock,
            net
        )

        vt_orderids: list = []

        for req in req_list:
            vt_orderid: str = self.main_engine.send_order(
                req, contract.gateway_name)

            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

            self.orderid_strategy_map[vt_orderid] = strategy

        return vt_orderids
    
    def send_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool,
        net: bool,
    ) -> list:
        """发送委托"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        if not contract:
            self.write_log(f"委托失败，找不到合约：{vt_symbol}", strategy)
            return ""

        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_volume)

        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=offset,
            type=OrderType.LIMIT,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{strategy.strategy_name}"
        )

        req_list: list[OrderRequest] = self.main_engine.convert_order_request(
            original_req,
            contract.gateway_name,
            lock,
            net
        )

        vt_orderids: list = []

        for req in req_list:
            vt_orderid: str = self.main_engine.send_order(
                req, contract.gateway_name)

            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

            self.orderid_strategy_map[vt_orderid] = strategy

        return vt_orderids

    def cancel_order(self, strategy: StrategyTemplate, vt_orderid: str) -> None:
        """委托撤单"""
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        if not order:
            self.write_log(f"撤单失败，找不到委托{vt_orderid}", strategy)
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_all(self, strategy: StrategyTemplate) -> None:
        """委托撤单"""
        for vt_orderid in list(strategy.active_orderids):
            self.cancel_order(strategy, vt_orderid)
        # for k,_ in strategy.symbol_status.items():
        #     strategy.symbol_status[k].is_active = False

    def get_engine_type(self) -> EngineType:
        """获取引擎类型"""
        return self.engine_type

    def get_pricetick(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """获取合约价格跳动"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        if contract:
            return contract.pricetick
        else:
            return None

    def get_size(self, strategy: StrategyTemplate, vt_symbol: str) -> int:
        """获取合约乘数"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        if contract:
            return contract.size
        else:
            return None

    def load_bars(self, strategy: StrategyTemplate, days: int, interval: Interval) -> None:
        """加载历史数据"""
        vt_symbols: list = strategy.vt_symbols
        dts: set[datetime] = set()
        history_data: dict[tuple, BarData] = {}

        # 通过接口、数据服务、数据库获取历史数据
        for vt_symbol in vt_symbols:
            data: list[BarData] = self.load_bar(vt_symbol, days, interval)

            for bar in data:
                dts.add(bar.datetime)
                history_data[(bar.datetime, vt_symbol)] = bar

        dts: list = list(dts)
        dts.sort()

        bars: dict = {}

        for dt in dts:
            for vt_symbol in vt_symbols:
                bar: Optional[BarData] = history_data.get((dt, vt_symbol), None)

                # 如果获取到合约指定时间的历史数据，缓存进bars字典
                if bar:
                    bars[vt_symbol] = bar
                # 如果获取不到，但bars字典中已有合约数据缓存, 使用之前的数据填充
                elif vt_symbol in bars:
                    old_bar: BarData = bars[vt_symbol]

                    bar = BarData(
                        symbol=old_bar.symbol,
                        exchange=old_bar.exchange,
                        datetime=dt,
                        open_price=old_bar.close_price,
                        high_price=old_bar.close_price,
                        low_price=old_bar.close_price,
                        close_price=old_bar.close_price,
                        gateway_name=old_bar.gateway_name
                    )
                    bars[vt_symbol] = bar

            self.call_strategy_func(strategy, strategy.on_bars, bars)

    def load_bar(self, vt_symbol: str, days: int, interval: Interval) -> list[BarData]:
        """加载单个合约历史数据"""
        symbol, exchange = extract_vt_symbol(vt_symbol)
        end: datetime = datetime.now(DB_TZ)
        start: datetime = end - timedelta(days)
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        data: list[BarData]

        # 通过接口获取历史数据
        if contract and contract.history_data:
            req: HistoryRequest = HistoryRequest(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start=start,
                end=end
            )
            data = self.main_engine.query_history(req, contract.gateway_name)

        # 通过数据服务获取历史数据
        else:
            data = self.query_bar_from_datafeed(symbol, exchange, interval, start, end)
            self.write_log("Because there is no datafeed, we can still try to read from database")

        # 通过数据库获取数据
        if not data:
            data = self.database.load_bar_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start=start,
                end=end,
            )

        return data

    def call_strategy_func(self, strategy: StrategyTemplate, func: Callable, params: object = None) -> None:
        """安全调用策略函数"""
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            strategy.trading = False
            strategy.inited = False

            msg: str = f"触发异常已停止\n{traceback.format_exc()}"
            self.write_log(msg, strategy)

    def add_strategy(
        self, class_name: str, strategy_name: str, vt_symbols: list, setting: dict
    ) -> None:
        """添加策略实例"""
        if strategy_name in self.strategies:
            self.write_log(f"创建策略失败，存在重名{strategy_name}")
            return

        strategy_class: Optional[StrategyTemplate] = self.classes.get(class_name, None)
        if not strategy_class:
            self.write_log(f"创建策略失败，找不到策略类{class_name}")
            return

        strategy: StrategyTemplate = strategy_class(self, strategy_name, vt_symbols, setting)
        self.strategies[strategy_name] = strategy

        for vt_symbol in vt_symbols:
            strategies: list = self.symbol_strategy_map[vt_symbol]
            strategies.append(strategy)

        self.save_strategy_setting()
        self.put_strategy_event(strategy)

    # def get_pos(self, strategy_name:str):
    #     self.dbservice.init_connection()
    #     pos_data = self.dbservice.select('current_pos', strategy = strategy_name)
    #     self.dbservice.close()
    #     return pos_data
        
    def init_strategy(self, strategy_name: str) -> None:
        """初始化策略"""
        self.init_executor.submit(self._init_strategy, strategy_name)

    def _init_strategy(self, strategy_name: str) -> None:
        """初始化策略"""
        strategy: StrategyTemplate = self.strategies[strategy_name]

        if strategy.inited:
            self.write_log(f"{strategy_name}已经完成初始化，禁止重复操作")
            return

        self.write_log(f"{strategy_name}开始执行初始化")

        # 调用策略on_init函数
        self.call_strategy_func(strategy, strategy.on_init)

        # 恢复策略状态
        # I make this redundant as it cannot reflect changes throughout time
        data: Optional[dict] = self.strategy_data.get(strategy_name, None)
        if data:
            for name in strategy.variables:
                value: Optional[object] = data.get(name, None)
                if value is None:
                    continue

                # # 对于持仓和目标数据字典，需要使用dict.update更新defaultdict
                # if name in {"pos_data", "target_data"}:
                #     strategy_data = getattr(strategy, name)
                #     strategy_data.update(value)
                # # 对于其他int/float/str/bool字段则可以直接赋值
                # else:
                #     setattr(strategy, name, value)
                if name not in {"pos_data", "target_data"}:
                    setattr(strategy, name, value)
                    
        # # my way of retrieving pos_data and target_data, please ignore the above from now
        # pos_data = self.get_pos(strategy_name=strategy_name)
        # for r in pos_data.iterrows():
        #     if r[1]['symbol'] in strategy.vt_symbols:
        #         strategy.set_pos(r[1]['symbol'], r[1]['pos'])

        # 订阅行情
        for vt_symbol in strategy.vt_symbols:
            contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
            if contract:
                req: SubscribeRequest = SubscribeRequest(
                    symbol=contract.symbol, exchange=contract.exchange)
                self.main_engine.subscribe(req, contract.gateway_name)
            else:
                self.write_log(f"行情订阅失败，找不到合约{vt_symbol}", strategy)

        # 推送策略事件通知初始化完成状态
        self.put_strategy_event(strategy)
        self.write_log(f"{strategy_name}初始化完成")
        strategy.inited = True

    def start_strategy(self, strategy_name: str) -> None:
        """启动策略"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        if not strategy.inited:
            self.write_log(f"策略{strategy.strategy_name}启动失败，请先初始化")
            return

        if strategy.trading:
            self.write_log(f"{strategy_name}已经启动，请勿重复操作")
            return

        # 调用策略on_start函数
        self.call_strategy_func(strategy, strategy.on_start)

        # 推送策略事件通知启动完成状态
        strategy.trading = True
        
        ## Might be unnecessary
        # self.dbservice.init_connection()
        # self.dbservice.update('strategies', "`status` = 'on'", strategy = strategy_name)
        # self.dbservice.close()
        self.put_strategy_event(strategy)

    def stop_strategy(self, strategy_name: str, message = None, header = None) -> None:
        """停止策略"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        if not strategy.trading:
            return
        
        if message is not None and type(message) == str:
            self.write_log(message)
        
        if message is not None and header is not None and type(message) == str and type(header) == str:
            try:
                self.email(header, message)
            except Exception as e:
                self.write_log(f"Failed to send email: {e}")

        # 调用策略on_stop函数
        self.call_strategy_func(strategy, strategy.on_stop)

        # 将交易状态设为False
        strategy.trading = False

        # 撤销全部委托
        self.cancel_all(strategy)

        # 同步数据状态
        self.sync_strategy_data(strategy)

        # 推送策略事件通知停止完成状态
        self.put_strategy_event(strategy)
        
        ## Might be unnecessary
        # self.dbservice.init_connection()
        # self.dbservice.update('strategies', "`status` = 'off'", strategy = strategy_name)
        # self.dbservice.close()

    def edit_strategy(self, strategy_name: str, setting: dict) -> None:
        """编辑策略参数"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        strategy.update_setting(setting)

        self.save_strategy_setting()
        self.put_strategy_event(strategy)

    def remove_strategy(self, strategy_name: str) -> bool:
        """移除策略实例"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        if strategy.trading:
            self.write_log(f"策略{strategy.strategy_name}移除失败，请先停止")
            return

        for vt_symbol in strategy.vt_symbols:
            strategies: list = self.symbol_strategy_map[vt_symbol]
            strategies.remove(strategy)

        for vt_orderid in strategy.active_orderids:
            if vt_orderid in self.orderid_strategy_map:
                self.orderid_strategy_map.pop(vt_orderid)

        self.strategies.pop(strategy_name)
        self.save_strategy_setting()

        self.strategy_data.pop(strategy_name, None)
        save_json(self.data_filename, self.strategy_data)

        return True

    def load_strategy_class(self) -> None:
        """加载策略类"""
        path1: Path = Path(__file__).parent.joinpath("strategies")
        self.load_strategy_class_from_folder(path1, "vnpy_portfoliostrategy.strategies")

        path2: Path = Path.cwd().joinpath("strategies")
        self.load_strategy_class_from_folder(path2, "strategies")

    def load_strategy_class_from_folder(self, path: Path, module_name: str = "") -> None:
        """通过指定文件夹加载策略类"""
        for suffix in ["py", "pyd", "so"]:
            pathname: str = str(path.joinpath(f"*.{suffix}"))
            for filepath in glob.glob(pathname):
                stem: str = Path(filepath).stem
                strategy_module_name: str = f"{module_name}.{stem}"
                self.load_strategy_class_from_module(strategy_module_name)

    def load_strategy_class_from_module(self, module_name: str) -> None:
        """通过策略文件加载策略类"""
        try:
            module: ModuleType = importlib.import_module(module_name)

            for name in dir(module):
                value = getattr(module, name)
                if (isinstance(value, type) and issubclass(value, StrategyTemplate) and value is not StrategyTemplate):
                    self.classes[value.__name__] = value
        except:  # noqa
            msg: str = f"策略文件{module_name}加载失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

    def load_strategy_data(self) -> None:
        """加载策略数据"""
        self.strategy_data = load_json(self.data_filename)

    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """保存策略数据到文件"""
        data: dict = strategy.get_variables()
        data.pop("inited")      # 不保存策略状态信息
        data.pop("trading")

        self.strategy_data[strategy.strategy_name] = data
        save_json(self.data_filename, self.strategy_data)

    def get_all_strategy_class_names(self) -> list:
        """获取所有加载策略类名"""
        return list(self.classes.keys())

    def get_strategy_class_parameters(self, class_name: str) -> dict:
        """获取策略类参数"""
        strategy_class: StrategyTemplate = self.classes[class_name]

        parameters: dict = {}
        for name in strategy_class.parameters:
            parameters[name] = getattr(strategy_class, name)

        return parameters

    def get_strategy_parameters(self, strategy_name) -> dict:
        """获取策略参数"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        return strategy.get_parameters()

    def init_all_strategies(self) -> None:
        """初始化所有策略"""
        for strategy_name in self.strategies.keys():
            self.init_strategy(strategy_name)

    def start_all_strategies(self) -> None:
        """启动所有策略"""
        for strategy_name in self.strategies.keys():
            self.start_strategy(strategy_name)

    def stop_all_strategies(self) -> None:
        """停止所有策略"""
        for strategy_name in self.strategies.keys():
            self.stop_strategy(strategy_name, 'stop ALL strategies')

    def load_strategy_setting(self) -> None:
        """加载策略配置"""
        strategy_setting: dict = load_json(self.setting_filename)

        for strategy_name, strategy_config in strategy_setting.items():
            self.add_strategy(
                strategy_config["class_name"],
                strategy_name,
                strategy_config["vt_symbols"],
                strategy_config["setting"]
            )

    def save_strategy_setting(self) -> None:
        """保存策略配置"""
        strategy_setting: dict = {}

        for name, strategy in self.strategies.items():
            strategy_setting[name] = {
                "class_name": strategy.__class__.__name__,
                "vt_symbols": strategy.vt_symbols,
                "setting": strategy.get_parameters()
            }

        save_json(self.setting_filename, strategy_setting)

    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """推送事件更新策略界面"""
        data: dict = strategy.get_data()
        event: Event = Event(EVENT_PORTFOLIO_STRATEGY, data)
        self.event_engine.put(event)

    def write_log(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """输出日志"""
        if strategy:
            msg: str = f"{strategy.strategy_name}: {msg}"

        log: LogData = LogData(msg=msg, gateway_name=APP_NAME)
        event: Event = Event(type=EVENT_PORTFOLIO_LOG, data=log)
        self.event_engine.put(event)
        
    def write_exception(self, msg: str, strategy: StrategyTemplate = None) -> None:
        self.main_engine.write_exception(msg, strategy)
        
    def email(self, title: str, msg: str) -> None:
        send_email(title, msg)

    def send_email(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """发送邮件"""
        if strategy:
            subject: str = f"{strategy.strategy_name}"
        else:
            subject: str = "组合策略引擎"

        self.main_engine.send_email(subject, msg)

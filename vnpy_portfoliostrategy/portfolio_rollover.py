from datetime import datetime
from time import sleep
from typing import TYPE_CHECKING, List, Optional
from copy import copy

from vnpy.trader.engine import MainEngine
from vnpy.trader.constant import OrderType
from vnpy.trader.object import ContractData, OrderRequest, SubscribeRequest, TickData
from vnpy.trader.object import Direction, Offset
from vnpy.trader.converter import OffsetConverter, PositionHolding

from vnpy_portfoliostrategy.engine import StrategyEngine, APP_NAME
from vnpy_portfoliostrategy.template import StrategyTemplate


class RolloverTool():
    """"""

    def __init__(self, ps_engine: StrategyEngine, main_engine: MainEngine, payup_spin:int = 5, max_volume_spin:int = 100) -> None:
        """"""
        super().__init__()
        
        self.ps_engine = ps_engine
        self.main_engine = main_engine
        self.old_symbol_list = []
        self.new_symbol_list = []
        self.payup_spin = payup_spin   # 5+
        self.max_volume_spin = max_volume_spin  # 1~10000 default 100
        self.rolled_list = []
        
    def init(self, strategy_name:str, old_symbol_list:list, new_symbol_list:list):
        self.strategy = self.ps_engine.strategies[strategy_name]
        self.old_symbol_list = old_symbol_list
        self.new_symbol_list = new_symbol_list

    def write_log(self, text: str) -> None:
        """"""
        now: datetime = datetime.now()
        text: str = now.strftime("%H:%M:%S\t") + text
        self.ps_engine.write_log(text)

    def subscribe(self, vt_symbol: str) -> None:
        """"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        if not contract:
            return

        req: SubscribeRequest = SubscribeRequest(contract.symbol, contract.exchange)
        self.main_engine.subscribe(req, contract.gateway_name)

    def roll_all(self) -> None:
        """"""
        # validation
        if len(self.old_symbol_list) == 0 or len(self.new_symbol_list) == 0 or\
            len(self.old_symbol_list) != len(self.new_symbol_list) or set(self.strategy.vt_symbols) != set(self.old_symbol_list):
            self.write_log(f"Incorrect rollover list!")
            return
        
        # examine whether new symbol is valid
        ### REMEMEBER: strategy needs to be init first otherwise won't be able to subscribe to market data!!!
        new_set:set = set(self.new_symbol_list) - set(self.old_symbol_list)
        for symb in new_set:
            self.subscribe(symb)
            sleep(1)
            new_tick: Optional[TickData] = self.main_engine.get_tick(symb)
            if not new_tick:
                self.write_log(f"无法获取目标合约{symb}的盘口数据，请先订阅行情")
                return
        payup: int = self.payup_spin

        # Roll position for each pair of old/new symbol
        for i in range(len(self.old_symbol_list)):
            old_symbol = self.old_symbol_list[i]
            new_symbol = self.new_symbol_list[i]
            self.roll_position(old_symbol, new_symbol, payup, self.strategy)
        self.rolled_list.append((self.old_symbol_list, self.new_symbol_list))


    def roll_position(self, old_symbol: str, new_symbol: str, payup: int, strategy: StrategyTemplate) -> None:
        """"""
        contract: ContractData = self.main_engine.get_contract(old_symbol)
        # converter: OffsetConverter = self.main_engine.get_converter(contract.gateway_name)
        # holding: PositionHolding = converter.get_position_holding(old_symbol)
        holding = strategy.get_pos(old_symbol)

        # Roll long position
        if holding>0:
            volume: float = holding

            self.send_order(
                old_symbol,
                Direction.SHORT,
                Offset.CLOSE,
                payup,
                volume
            )

            self.send_order(
                new_symbol,
                Direction.LONG,
                Offset.OPEN,
                payup,
                volume
            )

        # Roll short postiion
        if holding<0:
            volume: float = -holding

            self.send_order(
                old_symbol,
                Direction.LONG,
                Offset.CLOSE,
                payup,
                volume
            )

            self.send_order(
                new_symbol,
                Direction.SHORT,
                Offset.OPEN,
                payup,
                volume
            )


    def send_order(
        self,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        payup: int,
        volume: float,
    ) -> None:
        """
        Send a new order to server.
        """
        max_volume: int = self.max_volume_spin

        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        tick: Optional[TickData] = self.main_engine.get_tick(vt_symbol)

        if direction == Direction.LONG:
            price = tick.ask_price_1 + contract.pricetick * payup
        else:
            price = tick.bid_price_1 - contract.pricetick * payup

        while True:
            order_volume: int = min(volume, max_volume)

            original_req: OrderRequest = OrderRequest(
                symbol=contract.symbol,
                exchange=contract.exchange,
                direction=direction,
                offset=offset,
                type=OrderType.LIMIT,
                price=price,
                volume=order_volume,
                reference=f"{APP_NAME}_Rollover"
            )

            req_list: List[OrderRequest] = self.main_engine.convert_order_request(
                original_req,
                contract.gateway_name,
                False,
                False
            )

            vt_orderids: list = []
            for req in req_list:
                vt_orderid: str = self.main_engine.send_order(req, contract.gateway_name)
                if not vt_orderid:
                    continue

                vt_orderids.append(vt_orderid)
                self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

                msg: str = f"发出委托{vt_symbol}，{direction.value} {offset.value}，{volume}@{price}"
                self.write_log(msg)

            # Check whether all volume sent
            volume = volume - order_volume
            if not volume:
                break

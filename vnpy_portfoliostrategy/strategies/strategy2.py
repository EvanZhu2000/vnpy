from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine

class Strategy2(StrategyTemplate):
    # get trading hours
    # apply rebalancing rules
   
    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        for i in range(len(vt_symbols)):
            symb = vt_symbols[i]
            tar = setting['tarpos'][i]
            self.set_target(symb, tar)
    
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
        
    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        # tick.datetime
        if (self.get_target(tick.symbol) != self.get_pos(tick.symbol)):
            self.rebalance(tick.symbol, tick.ask_price_1, tick.bid_price_1, 'strategy2','rebalance')
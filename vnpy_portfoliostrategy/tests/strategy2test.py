from datetime import datetime
from importlib import reload
import vnpy_portfoliostrategy
reload(vnpy_portfoliostrategy)
from vnpy_portfoliostrategy import BacktestingEngine
from vnpy.trader.constant import Interval
import vnpy_portfoliostrategy.strategies.strategy2 as stg
reload(stg)
from vnpy_portfoliostrategy.strategies.strategy2 import Strategy2


# if __name__ == "__main__":
vt_symbols=["IH2406.CFFEX", "IH2409.CFFEX"]
engine = BacktestingEngine()
engine.set_parameters(
    vt_symbols=vt_symbols,
    interval=Interval.MINUTE,
    start=datetime(2024, 5, 7),
    end=datetime(2024, 6, 1),
    rates={
        vt_symbols[0]: 2/10000,
        vt_symbols[1]: 2/10000
    },
    slippages={
        vt_symbols[0]: 0,
        vt_symbols[1]: 0
    },
    sizes={
        vt_symbols[0]: 300,
        vt_symbols[1]: 300
    },
    priceticks={
        vt_symbols[0]: 0.2,
        vt_symbols[1]: 0.2
    },
    capital=10000000,
)
engine.init_strategy(Strategy2, setting)
engine.add_strategy(Strategy2, setting)
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
# engine.calculate_statistics()
# engine.show_chart()
# engine.get_all_trades(use_df=True).to_csv(r'C:\\Users\\Chris\\Desktop\\Evan\\trades.csv')
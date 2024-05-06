from datetime import datetime
from importlib import reload
import vnpy_portfoliostrategy
reload(vnpy_portfoliostrategy)
from vnpy_portfoliostrategy import BacktestingEngine
from vnpy.trader.constant import Interval
import vnpy_portfoliostrategy.strategies.strategy1 as stg
reload(stg)
from vnpy_portfoliostrategy.strategies.strategy1 import Strategy1


# if __name__ == "__main__":
engine = BacktestingEngine()
engine.set_parameters(
    vt_symbols=["IH2403.CFFEX", "IH2406.CFFEX"],
    interval=Interval.MINUTE,
    start=datetime(2024, 1, 2),
    end=datetime(2024, 3, 12),
    rates={
        "IH2403.CFFEX": 2/10000,
        "IH2406.CFFEX": 2/10000
    },
    slippages={
        "IH2403.CFFEX": 0,
        "IH2406.CFFEX": 0
    },
    sizes={
        "IH2403.CFFEX": 300,
        "IH2406.CFFEX": 300
    },
    priceticks={
        "IH2403.CFFEX": 0.2,
        "IH2406.CFFEX": 0.2
    },
    capital=2000000,
)
# setting = {
#     "boll_window": 20,
#     "boll_dev": 1,
# }
setting = {}
engine.add_strategy(Strategy1, setting)
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()
engine.get_all_trades(use_df=True).to_csv(r'C:\\Users\\Chris\\Desktop\\Evan\\trades.csv')
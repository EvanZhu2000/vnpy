from datetime import datetime
from typing import Optional
from vnpy.trader.object import  TickData

# An efficient implementation tracking whether all positions have been adjusted to desire
class BoolDict:
    def __init__(self, false_keys_list: list, target_time_dict: dict[str, datetime] = None):
        self.data = {}
        self.true_count = 0
        self.target_time_dict = target_time_dict # target time of execution of trades
        for k in false_keys_list:
            self.set(k, False)

    def set(self, key, value):
        # If the value is being changed from False to True
        if key not in self.data and value:
            self.true_count += 1
        elif key in self.data:
            if not self.data[key] and value:
                self.true_count += 1
            elif self.data[key] and not value:
                self.true_count -= 1
        self.data[key] = value

    def all_true(self):
        return self.true_count == len(self.data)
    
    def get_false_keys(self):
        return [key for key, value in self.data.items() if not value]
    
    
# class to track status for symbols
class SymbolStatus():
    is_active = False
    rej_counts = 0
    can_counts = 0
    order_list = []
    last_tick: Optional[TickData] = None  # for now it is just used to record whether we at least received one tick
    stop_FAK_cancel = False
    stop_late_tick_rebal = False
    
    def is_stop(self):
        return self.stop_FAK_cancel or self.stop_late_tick_rebal

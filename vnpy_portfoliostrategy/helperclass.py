from datetime import datetime
from typing import TYPE_CHECKING, Optional, DefaultDict, List, Union, Tuple
from vnpy.trader.constant import Interval, Direction, Offset
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
    alarm_time_since_last_tick = None
    
    def is_stop(self):
        return self.stop_FAK_cancel
    
class PositionInfo:
    """Position object that tracks both long and short positions separately"""
    
    def __init__(self, value: Union[int, "PositionInfo"] = 0) -> None:
        if isinstance(value, PositionInfo):
            self.long = value.long
            self.short = value.short
        else:
            # For single integer input, treat positive as long, negative as short
            self.long = max(value, 0)
            self.short = abs(min(value, 0))
            
    def __str__(self) -> str:
        return f"PositionInfo(long={self.long}, short={self.short})"
    
    @classmethod
    def from_long_short(cls, long: int = 0, short: int = 0) -> "PositionInfo":
        """Create a PositionInfo instance with specific long and short positions"""
        instance = cls(0)  # Create empty instance
        instance.long = long
        instance.short = abs(short)
        return instance
    
    def net_pos(self) -> int:
        """Return net position"""
        return self.long - self.short
        
    def minus(self, other: Union[int, "PositionInfo"]) -> List[Tuple[Direction, Offset, int]]:
        """
        Returns:
            actions: List of (Direction, Offset, volume) tuples representing required trades
        """
        if isinstance(other, int):
            other = PositionInfo(other)
            
        # Calculate differences in each direction
        long_diff = self.long - other.long
        short_diff = self.short - other.short
        
        actions = []
        
        # Handle long side changes
        if long_diff > 0:  # Need to reduce longs
            actions.append((Direction.SHORT, Offset.CLOSE, long_diff))
        elif long_diff < 0:  # Need to increase longs
            actions.append((Direction.LONG, Offset.OPEN, abs(long_diff)))
            
        # Handle short side changes    
        if short_diff > 0:  # Need to reduce shorts
            actions.append((Direction.LONG, Offset.CLOSE, short_diff))
        elif short_diff < 0:  # Need to increase shorts
            actions.append((Direction.SHORT, Offset.OPEN, abs(short_diff)))
            
        return actions

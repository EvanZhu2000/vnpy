import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

class RQ_API_MASKER:
    def __init__(self, rq=True) -> None:
        self.rq = rq
    
    def get_price(self):
        pass
    
    def get_member_rank(self, symb, start, end, rank_by):
        if self.rq:
            return futures.get_member_rank(symb,
                                            start_date=start,
                                            end_date=end, 
                                            rank_by=rank_by)
    
    def get_dominant(self, symb, start, end, rule, rank):
        if self.rq:
            return futures.get_dominant(symb, start,end,rule=rule,rank=rank)
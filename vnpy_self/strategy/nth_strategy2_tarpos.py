import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

from vnpy_self.pt_tools import *

from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()

today_date = datetime.today()

def retrieve_price(trading_list):
    _start = price_start
    _end = today_date.strftime('%Y%m%d')
    pr = get_price(trading_list, _start, _end, '1d')


def get_stats(trading_list, lookback_win_days):
    l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday= [pd.DataFrame()]*12

    for symb in tqdm(trading_list):
        _start = (today_date-pd.Timedelta(days==lookback_win_days)).strftime('%Y%m%d')
        _end = today_date.strftime('%Y%m%d')
        pr_df = get_price(f"{symb}888", _start,_end,'1d')

        df = futures.get_member_rank(symb,start_date=_start,end_date=_end, 
                                    rank_by='long')
        if df is None or pr_df is None:
            continue
            
        convert_df = df.reset_index()\
        .pivot(columns='member_name', index = 'trading_date',values='volume')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        l_df_everyday = pd.concat([l_df_everyday, convert_df],0)
        
        convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        l_df_delta_everyday = pd.concat([l_df_delta_everyday, convert_df],0)

        df = futures.get_member_rank(symb,start_date=_start,end_date=_end, 
                                    rank_by='short')
        convert_df = df.reset_index()\
        .pivot(columns='member_name', index = 'trading_date',values='volume')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        s_df_everyday = pd.concat([s_df_everyday, convert_df],0)
        
        convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        s_df_delta_everyday = pd.concat([s_df_delta_everyday, convert_df],0)
        
        dom_contracts = futures.get_dominant(symb, _start,_end,rule=0,rank=1)
        dom2_contracts = futures.get_dominant(symb, _start,_end,rule=0,rank=2)
        dom_contracts_schedule = pd.concat([dom_contracts.reset_index().groupby('dominant').first(),
                                            dom_contracts.reset_index().groupby('dominant').last()],1)
        dom_contracts_schedule.columns = ['f','l']
        dom2_contracts_schedule = pd.concat([dom2_contracts.reset_index().groupby('dominant').first(),
                                            dom2_contracts.reset_index().groupby('dominant').last()],1)
        dom2_contracts_schedule.columns = ['f','l']
        
        for i in dom_contracts_schedule.iterrows():
            # ========
            df = futures.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='long')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom_everyday = pd.concat([l_dom_everyday, convert_df],0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom_delta_everyday = pd.concat([l_dom_delta_everyday, convert_df],0)

            # ========
            df = futures.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='short')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom_everyday = pd.concat([s_dom_everyday, convert_df],0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom_delta_everyday = pd.concat([s_dom_delta_everyday, convert_df],0)
        
        for i in dom2_contracts_schedule.iterrows():
            # ========
            df = futures.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='long')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom2_everyday = pd.concat([l_dom2_everyday, convert_df],0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom2_delta_everyday = pd.concat([l_dom2_delta_everyday, convert_df],0)

            # ========
            df = futures.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='short')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom2_everyday = pd.concat([s_dom2_everyday, convert_df],0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom2_delta_everyday = pd.concat([s_dom2_delta_everyday, convert_df],0)
    return l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday

if __name__ == "__main__":

    lookback_win_days = 60
    price_start = pd.Timestamp('20240601')

    # 1. get potential trading list
    set1 = {'IF','IH','IC','IM','T','TS','TF','TL'}
    set2 = {'AG','AU','SC','EC','CU'}
    to_drop_list = list(set1|set2)
    potential_trading_list = []
    trading_list = potential_trading_list.drop(to_drop_list)

    # 2. get stats
    l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,\
        s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday = get_stats(trading_list, lookback_win_days)
    pro,pr88 = retrieve_price(trading_list)

    # 3. calculate signals
    xx = {
        '(-s_df.loc[:, symb,:]).diff().mean(1)':(1,20,5)
        ,'(-l_df.loc[:, symb,:]).diff().mean(1)':(1.4,20,5)
        ,'(-l_df.loc[:, symb,:]-s_df.loc[:, symb,:]).diff().mean(1)':(1.4,20,5)
        ,'(l_df.loc[:, symb,:]-s_df.loc[:, symb,:]).mean(1)':(1,20,5)
        ,'(l_df_delta.loc[:, symb,:]-s_df_delta.loc[:, symb,:]).mean(1)':(0.6,20,20)  
        ,'(-s_dom.loc[:, symb,:]).diff().mean(1)':(0.6,20,5)
        ,'(-l_dom.loc[:, symb,:]).diff().mean(1)':(1.4,5,20)
        ,'(-l_dom.loc[:, symb,:]-s_dom.loc[:, symb,:]).diff().mean(1)':(1,5,5)  
        ,'(l_dom.loc[:, symb,:]-s_dom.loc[:, symb,:]).mean(1)':(0.2,20,20)
        ,'(l_dom_delta.loc[:, symb,:]-s_dom_delta.loc[:, symb,:]).mean(1)':(0.2,20,10)
    }

    stat_list, result_list = [],[]
    for k,v in xx.items():
        stat = pd.DataFrame()
        for symb in tqdm(trading_list):
            if symb in l_df_everyday.index.get_level_values('symb').unique():
                stat = pd.concat([stat,eval(k).rename(symb)],1)   

        stat.index = pd.to_datetime(stat.index)
        stat_list.append(stat)
        result_list.append( bt_all(-settings_all([bband_para(stat,*v)]),
                            pro, 
                            pr88, 
                            mul_mappings,
                            mul_method = (price_start,pd.Timestamp('20240701')),
                            exec_delay=1,toRound=True))
    balancing_list = result_list[0][0].iloc[-1]

    # 4. insert balancing_list into database




    
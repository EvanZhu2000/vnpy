import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

from vnpy_self.pt_tools import *

from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()

from vnpy_self.strategy.rq_api_masker import RQ_API_MASKER
masker = RQ_API_MASKER()


def retrieve_price(trading_list):
    def get_clean_day_data(df,total_turnover_thres = 1e+8, open_interest_thres = 1000, volume_thres = 1000):
        df = df.copy()
        data = df.query(f"total_turnover>{total_turnover_thres} & open_interest>{open_interest_thres} & volume>{volume_thres}").reindex(df.index)

        data = data.unstack().stack(dropna=False)
        data_close = data['close'].unstack().T
        data_close = data_close.where(data_close[::-1].isna().cumsum()==0, np.nan)
        data.loc[data_close.T.stack(dropna=False).isna()] = np.nan
        data.dropna(inplace=True)
        data = data.sort_index()

        # calculation changes after 20200101 for some instruments
        if pd.Timestamp(df.index.get_level_values(1).min()) < pd.Timestamp('20200101'):
            if 'total_turnover' in data.columns:
                data.loc[(list(set(data.index.get_level_values(0).unique())-set(['IF','IH','IC','IM','T','TS','TF','TL'])),
                    slice(pd.Timestamp(df.index.get_level_values(1).min()),pd.Timestamp('20200101'))),
                    ['volume','total_turnover','open_interest']] /= 2
            else:
                data.loc[(list(set(data.index.get_level_values(0).unique())-set(['IF','IH','IC','IM','T','TS','TF','TL'])),
                    slice(pd.Timestamp(df.index.get_level_values(1).min()),pd.Timestamp('20200101'))),
                    ['volume','open_interest']] /= 2
        
        return data
    
    _start = price_start
    _end = today_date.strftime('%Y%m%d')

    data_raw = get_price((pd.Series(trading_list) + '888').tolist(), _start, _end, '1d')
    data_raw.set_index([data_raw.reset_index()['order_book_id'].str[:-3],data_raw.reset_index()['date']], inplace=True)
    data_1d = get_clean_day_data(data_raw)

    data_raw = get_price((pd.Series(trading_list) + '88').tolist(), _start, _end, '1d')
    data_raw.set_index([data_raw.reset_index()['order_book_id'].str[:-2],data_raw.reset_index()['date']], inplace=True)
    data_1d_ori = get_clean_day_data(data_raw)

    pro = data_1d['open'].unstack().T
    pr88 = data_1d_ori['close'].unstack().T
    return pro, pr88


def get_stats(trading_list, lookback_win_days, pro):
    l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday= [pd.DataFrame()]*12

    for symb in tqdm(trading_list):
        _start = (today_date-pd.Timedelta(lookback_win_days,'d')).strftime('%Y%m%d')
        _end = today_date.strftime('%Y%m%d')
        df = masker.get_member_rank(symb,start_date=_start,end_date=_end, 
                                    rank_by='long')
        if df is None or symb not in pro.columns or pro[symb] is None:
            continue
            
        convert_df = df.reset_index()\
        .pivot(columns='member_name', index = 'trading_date',values='volume')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        l_df_everyday = pd.concat([l_df_everyday, convert_df],axis=0)
        
        convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        l_df_delta_everyday = pd.concat([l_df_delta_everyday, convert_df],axis=0)

        df = masker.get_member_rank(symb,start_date=_start,end_date=_end, 
                                    rank_by='short')
        convert_df = df.reset_index()\
        .pivot(columns='member_name', index = 'trading_date',values='volume')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        s_df_everyday = pd.concat([s_df_everyday, convert_df],axis=0)
        
        convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
        convert_df['symb'] = symb
        convert_df = convert_df.set_index([convert_df.index,'symb'])
        s_df_delta_everyday = pd.concat([s_df_delta_everyday, convert_df],axis=0)
        
        dom_contracts = masker.get_dominant(symb, _start,_end,rule=0,rank=1)
        dom2_contracts = masker.get_dominant(symb, _start,_end,rule=0,rank=2)
        dom_contracts_schedule = pd.concat([dom_contracts.reset_index().groupby('dominant').first(),
                                            dom_contracts.reset_index().groupby('dominant').last()],axis=1)
        dom_contracts_schedule.columns = ['f','l']
        dom2_contracts_schedule = pd.concat([dom2_contracts.reset_index().groupby('dominant').first(),
                                            dom2_contracts.reset_index().groupby('dominant').last()],axis=1)
        dom2_contracts_schedule.columns = ['f','l']
        
        for i in dom_contracts_schedule.iterrows():
            # ========
            df = masker.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='long')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom_everyday = pd.concat([l_dom_everyday, convert_df],axis=0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom_delta_everyday = pd.concat([l_dom_delta_everyday, convert_df],axis=0)

            # ========
            df = masker.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='short')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom_everyday = pd.concat([s_dom_everyday, convert_df],axis=0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom_delta_everyday = pd.concat([s_dom_delta_everyday, convert_df],axis=0)
        
        for i in dom2_contracts_schedule.iterrows():
            # ========
            df = masker.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='long')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom2_everyday = pd.concat([l_dom2_everyday, convert_df],axis=0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                l_dom2_delta_everyday = pd.concat([l_dom2_delta_everyday, convert_df],axis=0)

            # ========
            df = masker.get_member_rank(i[0],start_date=i[1]['f'],end_date=i[1]['l'], 
                                        rank_by='short')
            if df is not None:
                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom2_everyday = pd.concat([s_dom2_everyday, convert_df],axis=0)

                convert_df = df.reset_index().pivot(columns='member_name', index = 'trading_date',values='volume_change')
                convert_df['symb'] = symb
                convert_df = convert_df.set_index([convert_df.index,'symb'])
                s_dom2_delta_everyday = pd.concat([s_dom2_delta_everyday, convert_df],axis=0)
    return l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday

if __name__ == "__main__":
    today_date = datetime.today()
    # today_date = datetime(2024, 9, 18)
    next_trading_date = get_next_trading_date(today_date)
    lookback_win_days = 60
    # price_start = pd.Timestamp('20240601')
    price_start = pd.Timestamp(today_date - pd.Timedelta(lookback_win_days,'d'))
    mul_mappings = mysqlservice.select('universe').set_index('root_symbol').to_dict()['multiplier']

    tmp1 = mysqlservice.select('strategies',date=next_trading_date,strategy = 'strategy2',status='on')
    if tmp1.shape[0]!=1:
        raise Exception('Wrong vnpy.strategies table for today!')
    initial_capital = float(tmp1['cash'][0]) * float(tmp1['leverage'][0])

    tmp2 = mysqlservice.select('trading_schedule', today = today_date.strftime('%Y-%m-%d'), strategy='strategy2')
    if tmp2.shape[0]!=1:
        raise Exception('Wrong vnpy.trading_schedule table for today!')
    if tmp1['date'][0]!=tmp2['date'][0]:
        raise Exception('Unmatching dates for vnpy.strategies & vnpy.trading_schedule')
    trading_list = (pd.Series(tmp2['symbol'][0].split(',')).str[:-4]).tolist()

    # 2. get stats
    pro,pr88 = retrieve_price(trading_list)
    l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,\
        s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday = get_stats(trading_list, lookback_win_days,pro)
        
    l_df,s_df,l_df_delta,s_df_delta,l_dom,s_dom,l_dom_delta,\
    s_dom_delta,l_dom2,s_dom2,l_dom2_delta,s_dom2_delta = l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,\
    s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday

    # 3. calculate signals
    xx = {'(-s_df.loc[:, symb,:]).diff().mean(1)': (1.8, 10, 20),
    '(-l_df.loc[:, symb,:]).diff().mean(1)': (1.0, 5, 10),
    '(-l_df.loc[:, symb,:]-s_df.loc[:, symb,:]).diff().mean(1)': (1.0, 20, 10),
    '(l_df.loc[:, symb,:]-s_df.loc[:, symb,:]).mean(1)': (1.0, 10, 20),
    '(l_df_delta.loc[:, symb,:]-s_df_delta.loc[:, symb ,:]).mean(1)': (0.6,
    20,
    5),
    '(-s_dom.loc[:, symb,:]).diff().mean(1)': (0.6, 20, 5),
    '(-l_dom.loc[:, symb,:]).diff().mean(1)': (0.6, 5, 20),
    '(-l_dom.loc[:, symb,:]-s_dom.loc[:, symb,:]).diff().mean(1)': (0.2, 20, 20),
    '(l_dom.loc[:, symb,:]-s_dom.loc[:, symb,:]).mean(1)': (1.8, 20, 20),
    '(l_dom_delta.loc[:, symb,:]-s_dom_delta.loc[:, symb,:]).mean(1)': (1.8,
    20,
  10)}

    stat_list, result_list = [],[]
    for k,v in xx.items():
        stat = pd.DataFrame()
        for symb in tqdm(trading_list):
            if symb in l_df_everyday.index.get_level_values('symb').unique():
                stat = pd.concat([stat,eval(k).rename(symb)],axis=1)   

        stat.index = pd.to_datetime(stat.index)
        stat_list.append(stat)
    
    g = weight_cap(-settings_all([bband_para(stat_list[5],0.6, 20, 5)]), mul_mappings, pr88, initial_capital=1000000)
    balancing_list = g.replace(np.nan,0).iloc[-1]

    # 4. insert balancing_list into database
    if today_date.date() != balancing_list.name.date():
        raise Exception(f'Wrong tar pos date! {today_date}, {balancing_list.name}')
    mysqlservice.insert("daily_rebalance_target", date=next_trading_date, today=today_date,
        symbol = ','.join(balancing_list.astype(int).astype(str).index), 
        target = ','.join(balancing_list.astype(int).astype(str).values),
        strategy = 'strategy2')
    mysqlservice.close()

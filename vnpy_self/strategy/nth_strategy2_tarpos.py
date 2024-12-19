import rqdatac as rq
from rqdatac import *
rq.init('+85260983439','evan@cash')

from vnpy_self.pt_tools import *
import sys

from vnpy_portfoliostrategy.mysqlservice import MysqlService
mysqlservice = MysqlService()
mysqlservice.init_connection()

from vnpy_self.strategy.rq_api_masker import RQ_API_MASKER
masker = RQ_API_MASKER()

def symbol_rq2vnpy(l, all_data):
    return all_data.loc[all_data['underlying_symbol'].isin(l)][['trading_code','exchange']].apply(lambda x: '.'.join(x), axis = 1).values

def retrieve_price(trading_list, price_start, today_date):
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
    pr = data_1d['close'].unstack().T
    pr88 = data_1d_ori['close'].unstack().T
    return pro, pr, pr88


def get_stats(trading_list, lookback_days, pro, today_date):
    l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday= [pd.DataFrame()]*12

    for symb in tqdm(trading_list):
        _start = (today_date-pd.Timedelta(lookback_days,'d')).strftime('%Y%m%d')
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

def run(today_date_str:str): 
    all_data = all_instruments(type='Future', market='cn', date=None)
    xxx = {
    'x0':('(-s_dom.loc[:, symb,:]).diff().mean(1)',(1.4,5,5)),
    'x1':('(-s_dom.loc[:, symb,:]).diff().mean(1)',(1.4,20,5)),
    'x2':('(-s_dom.loc[:, symb,:]).diff().mean(1)',(1.4,10,5)),
    'x3':('(-s_df.loc[:, symb,:]).diff().mean(1)',(1.8, 10, 60)),
    'x4':('(-s_df.loc[:, symb,:]).diff().mean(1)',(1.7, 45, 5)),
    'x5':('(-s_df.loc[:, symb,:]).diff().mean(1)',(1.7, 30, 5)),
    }
        
    today_date = datetime.strptime(today_date_str, "%Y-%m-%d")
    # today_date = datetime(2024, 9, 18)
    next_trading_date = get_next_trading_date(today_date)
    trading_dates = pd.to_datetime(get_trading_dates(start_date='20150105', end_date=today_date))
    max_para = max(list(sum(list(sum(xxx.values(), ()))[1::2], ())))
    lookback_days = (today_date - trading_dates[-int(2.1*max_para):][0]).days
    price_start = pd.Timestamp(today_date - pd.Timedelta(lookback_days,'d'))
    mul_mappings = mysqlservice.select('universe').set_index('root_symbol').to_dict()['multiplier']

    tmp1 = mysqlservice.select('strategies',date=next_trading_date,strategy = 'strategy2',status='on')
    if tmp1.shape[0]!=1:
        raise Exception('Wrong vnpy.strategies table for today!')
    money = float(tmp1['cash'][0]) * float(tmp1['leverage'][0])

    tmp2 = mysqlservice.select('trading_schedule', today = today_date.strftime('%Y-%m-%d'), strategy='strategy2')
    if tmp2.shape[0]!=1:
        raise Exception('Wrong vnpy.trading_schedule table for today!')
    if tmp1['date'][0]!=tmp2['date'][0]:
        raise Exception('Unmatching dates for vnpy.strategies & vnpy.trading_schedule')
    trading_list = id_convert(tmp2['symbol'][0].split(','))

    # 2. get stats
    pro,pr,pr88 = retrieve_price(trading_list, price_start, today_date)
    l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,\
        s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday = get_stats(trading_list, lookback_days, pro, today_date)
        
    l_df,s_df,l_df_delta,s_df_delta,l_dom,s_dom,l_dom_delta,\
    s_dom_delta,l_dom2,s_dom2,l_dom2_delta,s_dom2_delta = l_df_everyday,s_df_everyday,l_df_delta_everyday,s_df_delta_everyday,l_dom_everyday,s_dom_everyday,l_dom_delta_everyday,\
    s_dom_delta_everyday,l_dom2_everyday,s_dom2_everyday,l_dom2_delta_everyday,s_dom2_delta_everyday

    # 3. calculate signals
    stat_list,set_list = [],[]
    for k,v in xxx.items():
        stat = pd.DataFrame()
        for symb in tqdm(trading_list):
            if symb in l_df.index.get_level_values('symb').unique():
                stat = pd.concat([stat,eval(v[0]).rename(symb).sort_index()],axis=1)   

        stat.index = pd.to_datetime(stat.index)
        stat.sort_index(axis=1,inplace=True)
        stat.sort_index(axis=0,inplace=True)
        stat_list.append(stat)
        set_list.append(bband_para(stat,*v[1]))
    
    sam = sampler(trading_dates,'20150105',samp_days=20)
    used_money = 1000000
    w1 = weight(-settings_all(set_list,'x3&x4','x3&x4'), mul_mappings, pr88, sam, initial_capital=money, toRound=False, used_cap_limit=used_money)
    w2 = weight(-settings_all(set_list,'x0&x1&x3&x4','x0&x1&x3&x4'), mul_mappings, pr88, sam, initial_capital=money, toRound=False, used_cap_limit=used_money)
    w3 = weight(-settings_all(set_list,'x0&x3','x0&x3'), mul_mappings, pr88, sam, initial_capital=money, toRound=False, used_cap_limit=used_money)
    w = -(0.5*w1[0]+0.3*w2[0]+0.2*w3[0]).round(0)
    d,m = bt_all(w,
             pro[w.columns],
            pr88[w.columns],
              pr[w.columns],
            mul_mappings,
            exec_delay=1,
            initial_capital=used_money,
            commission=0.0001,toFormat=True)
    d.to_csv('~/miniconda3/envs/vnpy3/lib/python3.10/site-packages/vnpy_self/analysis/data/strategy2_expected_pnl.csv')
    w.to_csv('~/miniconda3/envs/vnpy3/lib/python3.10/site-packages/vnpy_self/analysis/data/strategy2_positions.csv')
    balancing_list = w.replace(np.nan,0).iloc[-1]

    # 4. insert balancing_list into database
    if today_date.date() != balancing_list.name.date():
        raise Exception(f'Wrong tar pos date! {today_date}, {balancing_list.name}')
    mysqlservice.insert("daily_rebalance_target", date=next_trading_date, today=today_date,
        symbol = ','.join(symbol_rq2vnpy(balancing_list.astype(int).astype(str).index, all_data)), 
        target = ','.join(balancing_list.astype(int).astype(str).values),
        strategy = 'strategy2')
    mysqlservice.close()
    
if __name__ == "__main__":
    # The input today_date needs to be the real date at next settlement date start, in the format of YYYY-MM-DD
    today_date_str = sys.argv[1]
    run(today_date_str)
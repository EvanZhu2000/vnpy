import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import tools
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
from tqdm import tqdm

from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from scipy import stats
from scipy.stats import norm

from itertools import product
from itertools import chain
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import product
import itertools

import re
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")


# could have some problems
from ds_tools import *
from constants import *

def format_number(n):
    return re.sub(r"(\d)(?=(\d{4})+(?!\d))", r"\1,", str(n))

def get_rq_specific_time_df(df, time_slice):
    tmp = df.swaplevel().unstack().between_time(time_slice[0], time_slice[-1]).stack().swaplevel().sort_index()
    tmp.index = pd.MultiIndex.from_arrays([tmp.index.get_level_values(0),
                                      pd.to_datetime(tmp.index.get_level_values(1)).date],
                                     names = ['order_book_id', 'datetime'])
    return tmp


def cleaner(df, start='09:35:00', end='15:54:59'):
    df = df[df.index.dayofweek < 5]
    df = df.iloc[df.index.indexer_between_time(start, end)]
    df = df.iloc[df.index.indexer_between_time('13:01:00', '11:58:59')]
    return df


# slice by seconds
def slicer(df, win):
    return df.reset_index().groupby(df.reset_index().index // win).last().set_index('index')


def getReturn(df, win=1):
    ret = cleaner(df / df.shift(win, freq='s') - 1)
    return slicer(ret, win)[0]


def getDiff(df, win=1):
    ret = cleaner(df - df.shift(win, freq='s'))
    return slicer(ret, win)[0]


def getPrice(df, name, start_date, end_date, freq='1min', win=1):
    res = slicer(cleaner(df), win)
    tmp_grouper = res.reset_index().groupby(pd.Grouper(key='index', freq=freq))
    res = pd.concat([tmp_grouper.first()['open'],
                     tmp_grouper.max()['high'],
                     tmp_grouper.min()['low'],
                     tmp_grouper.last()['close'],
                     tmp_grouper.sum()['volume'],
                     tmp_grouper.last()['P_1_B'],
                     tmp_grouper.last()['Q_1_B'],
                     tmp_grouper.last()['P_1_A'],
                     tmp_grouper.last()['Q_1_A']], axis=1,
                    keys=['open', 'high', 'low', 'close', 'volume', 'P_1_B', 'Q_1_B', 'P_1_A', 'Q_1_A']).replace(0,
                                                                                                                 np.nan).dropna(
        how='all')
    res = res.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if freq[-1] == 'd':
        res.index = pd.MultiIndex.from_product([[name], res.index.tolist()], names=['order_book_id', 'date'])
    else:
        res.index = pd.MultiIndex.from_product([[name], res.index.tolist()], names=['order_book_id', 'datetime'])

    return res

def adf_test(df):
    if df.dropna().shape[0]>100000:
        print("Too long series")
    else:
        test_stat = adfuller(df.dropna())[1]
        print(f"Series is{'' if test_stat<0.05 else ' not'} stationary, val = {test_stat:.4f}, thes = 0.05")

def cal_regression(df1,df2):
    x = df1.values.reshape(-1,1)
    y = df2.values.reshape(-1,1)
    reg = sm.OLS(y, x).fit()
    return -reg.params[0]


def bt_all(g_df, ins_price, ins_price_to_cal_fee, mul_map, 
           to=None, mul_method = 60, initial_capital=10000000, commission = 0.0001, exec_delay=0, toFormat=True,toRound=False):
    '''
    all of g_df, ins_price and ins_price_to_cal_fee need to be dataframe with columns as sequenced datetimes and index as names
    ins_price_to_cal_fee: the price to calculate fees, used in pairs trading, usually need to be 88
    mul: dictionary
    
    Explanation:
    to: if provided with turnover, can calculate maximum capacity
    initial_capital: initial_capital for each instrument
    turnover: -1 to 1 (or the opposite) = 100%, to 0 means 50%
    trades_count: open and close means one trade, the output is annual trade per instrument
    '''
    print(to)
    # validation
    if not (type(g_df) == pd.core.frame.DataFrame) \
    or not (type(ins_price) == pd.core.frame.DataFrame) \
    or not (type(ins_price_to_cal_fee) == pd.core.frame.DataFrame):
        raise Exception('bad input format!')
    
    ins_price = ins_price.shift(-exec_delay).copy()
    ins_price_to_cal_fee = ins_price_to_cal_fee.shift(-exec_delay).copy()
    if exec_delay>0:
        ins_price = ins_price.iloc[:-exec_delay]
        ins_price_to_cal_fee = ins_price_to_cal_fee.iloc[:-exec_delay]
    if not ins_price.index.equals(ins_price_to_cal_fee.index) or not ins_price.columns.equals(ins_price_to_cal_fee.columns):
        raise Exception('unmatching ins_price & ins_price_to_cal_fee')
    
    intersec_index = g_df.index.intersection(ins_price.index)
    g_df = g_df.loc[intersec_index].copy()
    ins_price = ins_price.loc[intersec_index]
    ins_price_to_cal_fee = ins_price_to_cal_fee.loc[intersec_index]
    if len(set(g_df.index) - set(ins_price.index))!=0:
        raise Exception('no price for some datetime')
    if len(set(g_df.columns) - set(ins_price.columns))!=0:
        raise Exception('no price for some symbols')
    if len(set(g_df.columns) - set(mul_map.keys()))!=0:
        raise Exception('no multiplier for some symbols')

    # make the dataframes match with each other
    ins_price = ins_price[g_df.columns]
    ins_price_to_cal_fee = ins_price_to_cal_fee[g_df.columns]
    g_df = g_df.reindex(ins_price.index).ffill()
    mul_series = pd.Series(mul_map)[g_df.columns]
    
    # statistics before equal-weighting
    c_df = g_df * mul_series * ins_price
    trades_count = (g_df.diff().replace(0,np.nan).abs().sum() / 2).mean()
    turnover = (g_df/g_df.shape[1]).diff().abs().sum(1).mean() / 2
    mulprice_fee = (mul_series * ins_price_to_cal_fee).abs()
    recent_start, recent_end = mulprice_fee.index[-1] - pd.Timedelta(days=60), mulprice_fee.index[-1]
    
    # calculate equal turnover portfolio
    if mul_method:
        if mul_method == 'mean':
            multiplier = initial_capital / mulprice_fee.describe().loc['mean']
        elif mul_method == 'max':
            multiplier = initial_capital / mulprice_fee.describe().loc['max']
        elif mul_method == 'first':
            multiplier = initial_capital / mulprice_fee.iloc[0]
        elif mul_method == 'last':
            multiplier = initial_capital / mulprice_fee.iloc[-1]
        elif type(mul_method) == int:
            recent_start, recent_end = mulprice_fee.index[-1] - pd.Timedelta(days=mul_method), mulprice_fee.index[-1]
            multiplier = initial_capital / mulprice_fee.loc[recent_start:recent_end].describe().loc['mean']
        elif type(mul_method) == tuple:
            recent_start, recent_end = mul_method
            multiplier = initial_capital / mulprice_fee.loc[recent_start:recent_end].describe().loc['mean']
        else:
            raise Exception('wrong mul_method!')
        g_df *= multiplier
        if toRound:
            g_df = g_df.round(0)
        c_df = g_df * mul_series * ins_price
        
    # statistics after equal-weighting
    max_divisor = g_df.abs().stack().replace(0,np.nan).dropna().describe().loc['25%']
    min_cash_needed = (c_df.abs().sum(1).loc[recent_start:recent_end]).mean()/max_divisor
    ttl_capacity = 0
    print(to)
    if to is not None:
        ttl_capacity = min_cash_needed * max_divisor * (to.mean() / initial_capital).describe().loc['50%'] / 1000000000
    
    # calculate commission
    comm_df = commission * g_df.diff().abs() * mulprice_fee
    
    # get daily settled pnl
    price_daily = ins_price.groupby(ins_price.index.date).last()
    g_daily = g_df.groupby(g_df.index.date).last()
    c_daily = c_df.groupby(c_df.index.date).last()
    comm_daily = comm_df.groupby(comm_df.index.date).sum()
    daily_pnl = ((g_daily.diff()*mul_series*price_daily - comm_daily).cumsum().reindex(price_daily.index).ffill() - c_daily).diff()
    
    # summary of the backtest
    result = []
    mean_daily_pnl = daily_pnl.mean(1)
    if toFormat:
        result.append("%.4f"%calculate_sharpe_ratio(mean_daily_pnl))
        result.append("%.2f"%(100*calculate_annual_return(mean_daily_pnl, initial_capital))+'%')
        result.append("%.2f"%(100*calculate_max_drawdown(mean_daily_pnl, initial_capital))+'%')
        result.append("%.2f"%(100*turnover)+'%')
        result.append("%.2f"%(trades_count / effective_year_count(mean_daily_pnl)))
        result.append("%.2f"%(effective_year_count(mean_daily_pnl)) + ' yrs')
        result.append(format_number(int(min_cash_needed)))
        result.append(str(int(ttl_capacity)) + ' b')
    else:
        result.append(calculate_sharpe_ratio(mean_daily_pnl))
        result.append(calculate_annual_return(mean_daily_pnl, initial_capital))
        result.append(calculate_max_drawdown(mean_daily_pnl, initial_capital))
        result.append(turnover)
        result.append((trades_count / effective_year_count(mean_daily_pnl)))
        result.append((effective_year_count(mean_daily_pnl)))
        result.append(min_cash_needed)
        result.append(ttl_capacity)
    summary_df = pd.DataFrame(result, index=['sharpe','annu_ret','mdd','turnover','#trades','period','min_cash','ttl_capcity']).T
    return g_df, daily_pnl, summary_df


def bt(a_df, ins_price, ins_price_to_cal_fee, mul, mul_method = None, initial_capital=None, commission = 0.0003, exec_delay=0, conform=True):
    '''
    ins_price_to_cal_fee: the price to calculate fees, used in pairs trading
    '''
    if not ((type(a_df) == pd.core.frame.DataFrame) & (0 in a_df.columns) & (len(a_df.columns) == 1)):
        raise Exception('bad a_df format!')
    ins_price,ins_price_to_cal_fee = ins_price.copy(),ins_price_to_cal_fee.copy()
    ins_price.name,ins_price_to_cal_fee.name = None,None
    ins_price.index.name,ins_price_to_cal_fee.index.name = None,None
    ins_price = ins_price.shift(-exec_delay).dropna()
    ins_price_to_cal_fee = ins_price_to_cal_fee.shift(-exec_delay).dropna()
    a_df = a_df.loc[a_df.index.intersection(ins_price.index)].sort_index()
    order = ins_price.loc[a_df.index]
    order *= a_df[0] * mul
    order = pd.concat([a_df, order, order-commission*ins_price_to_cal_fee.loc[a_df.index]*mul],axis=1)
    order.columns = ['q','ba_ins','ba_ins_fees']
    if order.shape[0] % 2!=0:
        order = order.iloc[:-1].copy()
    to_mul_price = (ins_price_to_cal_fee.loc[order.index]*mul).abs()
    if mul_method:
        if mul_method == 'mean':
            order *= initial_capital / to_mul_price.describe()['mean']
        elif mul_method == 'max':
            order *= initial_capital / to_mul_price.describe()['max']
        elif mul_method == 'first':
            order *= initial_capital / to_mul_price.iloc[0]
        elif mul_method == 'last':
            order *= initial_capital / to_mul_price.iloc[-1]
        else:
            raise Exception('wrong mul_method!')
    
    # trade pnl
    tmp = order.reset_index().groupby(order.reset_index().index//2)
    start_time = tmp.first()['index']
    end_time = tmp.last()['index']
    pnl = order.groupby(order.reset_index().index//2).sum()[['ba_ins','ba_ins_fees']]
    trade_pnl = pd.concat([start_time, end_time, pnl], axis=1)
    trade_pnl.columns = ['start_time','end_time','pnl_net_fees','pnl']
    
    # daily pnl
    _order = order.sort_index()
    qwe = pd.DataFrame(np.nan,
                       index=ins_price.reset_index().groupby(ins_price.index.date).last().index,
                       columns=['q','ba_ins_fees'])
    qwe.loc[_order['q'].to_frame().set_index(_order.index.date).index, 'q'] = -(_order['q'].cumsum().values)
    qwe.loc[_order['ba_ins_fees'].to_frame().set_index(_order.index.date).index, 'ba_ins_fees'] = _order['ba_ins_fees'].cumsum().values
    qwe = qwe.ffill().replace(np.nan, 0)
    qwe.set_index(ins_price.reset_index().groupby(ins_price.index.date).last()['index'], inplace=True)
    qwe['q'] = qwe['q'] * ins_price.reset_index().groupby(ins_price.index.date).last().set_index('index')[0] * mul
    daily_pnl = qwe.eval('ba_ins_fees+q').diff()
        
    # metrics
    if initial_capital is None:
        initial_capital = o['ba_ins_fees'].abs().max()
    metrics = get_metrics(order, trade_pnl, daily_pnl, initial_capital, risk_free_rate=0)
    if conform==True and order.groupby(order.reset_index().index//2).sum()['q'].sum() != 0:
        starting_ind = order.groupby(order.reset_index().index//2).sum()['q'].loc[order.groupby(order.reset_index().index//2).sum()['q']!=0].index[0]
        raise Exception(f"Orders don't conform\n {order.iloc[2*starting_ind:2*starting_ind+2]}")
    return order,trade_pnl,daily_pnl,metrics



def bt_raw(ttl_cash, ins1_df, ins2_df, mul1, mul2, a_df, h_df,
           r=0, lev=1, commission=0.0001, shortsell=0.0001,
           bidask=True, exec_delay=0):
    '''
    both ins1_df and ins2_df are price and quantity dataframe
    shortsell per transaction
    '''
    ins1_df = ins1_df.shift(-exec_delay).copy()
    ins2_df = ins2_df.shift(-exec_delay).copy()
    tmp_ins1 = ins1_df.loc[a_df.index].astype(float)
    tmp_ins2 = ins2_df.loc[a_df.index].astype(float)
    if bidask:
        tmp_ins1[0] = tmp_ins1.eval('(P_1_B + P_1_A)/2')
        tmp_ins2[0] = tmp_ins2.eval('(P_1_B + P_1_A)/2')
    tmp = pd.concat([h_df.squeeze() * tmp_ins1[0] + tmp_ins2[0],
                     a_df, h_df], axis=1)

    # calculate quantity
    ratio = pd.concat([h_df.reset_index(), pd.Series([1] * h_df.shape[0])], axis=1).drop(columns='index')
    ratio.columns = ['ins1', 'ins2']
    q1 = ratio.abs().eval('ins1/(ins1 + ins2)') / mul1
    q2 = ratio.abs().eval('ins2/(ins1 + ins2)') / mul2
    max_mul = ttl_cash / (abs(q1 * mul1 * tmp_ins1.dropna()[0].iloc[0]) + abs(q2 * mul2 * tmp_ins2.dropna()[0].iloc[0]))
    q1, q2 = q1 * max_mul, q2 * max_mul
    q1, q2 = q1 * lev / 2, q2 * lev / 2
    if r is not None:
        q1 = round(q1, r)
        q2 = round(q2, r)
    if q1.all() == 0 or q2.all() == 0:
        raise Exception('Not enough money!')
    quantity = pd.concat([-q1, q2], axis=1)
    quantity.columns = ['ins1', 'ins2']
    tmp = pd.concat([tmp, quantity.set_index(tmp.index)], axis=1)
    tmp.columns = ['sig', 'act', 'h', 'q1', 'q2']

    # buy means quantity is positive
    tmp['q1'] *= tmp['act']
    tmp['q2'] *= tmp['act']
    tmp['ba_ins1'] = tmp_ins1[0] * tmp['q1'].values * mul1
    tmp['ba_ins2'] = tmp_ins2[0] * tmp['q2'].values * mul2
    if bidask:
        tmp['ba_ins1_spread'] = tmp_ins1['P_1_B'].where(tmp['q1'] > 0, tmp_ins1['P_1_A']) * tmp['q1'].values * mul1
        tmp['ba_ins2_spread'] = tmp_ins2['P_1_B'].where(tmp['q2'] > 0, tmp_ins2['P_1_A']) * tmp['q2'].values * mul2
    else:
        tmp['ba_ins1_spread'] = tmp['ba_ins1']
        tmp['ba_ins2_spread'] = tmp['ba_ins2']

        # calcuate performance after fees
    fees = pd.Series([(1 + commission)] * a_df.shape[0], index=a_df.index)
    tmp['ba_ins1_fees'] = tmp['ba_ins1_spread'].squeeze() * fees.where((tmp['ba_ins1_spread'] < 0).squeeze(),
                                                                       (1 - commission - shortsell))
    tmp['ba_ins2_fees'] = tmp['ba_ins2_spread'].squeeze() * fees.where((tmp['ba_ins2_spread'] < 0).squeeze(),
                                                                       (1 - commission - shortsell))
    
    if bidask:
        tt,dd = get_pnl_from_orders(tmp, ins1_df['TP'].to_frame(), ins2_df['TP'].to_frame(), mul1, mul2)
    else:
        tt,dd = get_pnl_from_orders(tmp, ins1_df, ins2_df, mul1, mul2)
    mm = get_metrics(tmp, tt, dd, ttl_cash)
    return tmp, tt, dd, mm


def get_pnl_from_orders(order, ins1_df, ins2_df, mul1, mul2):
    # trade pnl
    tmp = order.eval('ba_ins1_fees+ba_ins2_fees').reset_index().groupby(order.reset_index().index // 2)
    start_time = tmp.first()['index']
    end_time = tmp.last()['index']
    pnl = order.eval('ba_ins1_fees+ba_ins2_fees').groupby(order.reset_index().index // 2).sum()
    theory_pnl = order.eval('ba_ins1+ba_ins2').groupby(order.reset_index().index // 2).sum()
    net_fees_pnl = order.eval('ba_ins1_spread+ba_ins2_spread').groupby(order.reset_index().index // 2).sum()
    res = pd.concat([start_time, end_time, pnl, net_fees_pnl, theory_pnl], axis=1)
    res.columns = ['start_time', 'end_time', 'pnl', 'net_fees_pnl', 'theory_pnl']
    res['sig_infer'] = order.eval('sig*act').groupby(order.reset_index().index // 2).sum()

    # daily pnl
    _order = order.sort_index()
    qwe = pd.DataFrame(np.nan,
                       index=ins1_df.reset_index().groupby(ins1_df.index.date).last().index,
                       columns=['q1', 'q2', 'ba_ins1_fees', 'ba_ins2_fees'])
    qwe.loc[_order[['q1', 'q2']].set_index(_order.index.date).index, ['q1', 'q2']] = -(
        _order[['q1', 'q2']].cumsum()).set_index(_order.index.date)
    qwe.loc[
        _order[['ba_ins1_fees', 'ba_ins2_fees']].set_index(_order.index.date).index, ['ba_ins1_fees', 'ba_ins2_fees']] = \
    _order[['ba_ins1_fees', 'ba_ins2_fees']].cumsum().set_index(_order.index.date)
    qwe = qwe.ffill().replace(np.nan, 0)
    qwe.set_index(ins1_df.reset_index().groupby(ins1_df.index.date).last()['index'], inplace=True)
    qwe['q1'] = qwe['q1'] * ins1_df.reset_index().groupby(ins1_df.index.date).last().set_index('index').squeeze() * mul1
    qwe['q2'] = qwe['q2'] * ins2_df.reset_index().groupby(ins2_df.index.date).last().set_index('index').squeeze() * mul2
    daily_pnl = qwe.eval('ba_ins1_fees+q1 + ba_ins2_fees+q2').diff()
    return res, daily_pnl

def effective_year_count(pnl_series):
    return ((pnl_series.index[-1] - pnl_series.index[0]).days / 365)

def calculate_cumulative_return(pnl_series, initial_capital):
    return pnl_series.sum() / initial_capital

def calculate_annual_return(pnl_series,initial_capital):
    return calculate_cumulative_return(pnl_series, initial_capital) / effective_year_count(pnl_series)

def calculate_sharpe_ratio(pnl_series, days=250):
    return np.mean(pnl_series) / (np.std(pnl_series)+1e-6) * math.sqrt(days)

def calculate_rolling_sharpe(pnl_series, days=250):
     return pnl_series.rolling(days).apply(calculate_sharpe_ratio,args=(days,),raw=True,engine='numba')
#     return pnl_series.rolling('365D', min_periods = 220).apply(calculate_sharpe_ratio,raw=True)

def calculate_win_ratio(pnl_series):
    pl = pnl_series.replace(0, np.nan).dropna()
    win_ratio = len(pl[pl > 0]) / len(pl)
    return win_ratio

def calculate_profit_loss_ratio(pnl_series):
    total_profit = pnl_series[pnl_series > 0].sum()
    total_loss = pnl_series[pnl_series < 0].sum()
    profit_loss_ratio = abs(total_profit / total_loss)
    return profit_loss_ratio

def calculate_drawdown(pnl_series, initial_capital):
    cumulative_returns = (initial_capital + pnl_series.cumsum()) / initial_capital
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

def calculate_max_drawdown(pnl_series, initial_capital):
    max_drawdown = calculate_drawdown(pnl_series, initial_capital).min()
    return max_drawdown


def calculate_trade_time(pnl_df):
    if pnl_df['end_time'].dt.hour.sum() == 0:
        precision = 'days'
        divider = 60*60*24
    elif pnl_df['end_time'].dt.minute.sum() == 0:
        precision = 'hours'
        divider = 60*60
    elif pnl_df['end_time'].dt.second.sum() == 0:
        precision = 'minutes'
        divider = 60
    else:
        precision = 'seconds'
    return (((pnl_df['end_time'] - pnl_df['start_time']).dt.total_seconds()/divider).describe().loc[['min', '25%', '50%', '75%', 'max']].astype(int).astype(str)+' '+precision).values

def get_metrics_title():
    return ["count","annu_count","cumu_ret","annu_ret","annu_sharpe","mdd","win","p_l",'t_min', 't_25%', 't_50%', 't_75%', 't_max']

def get_metrics(order_df, pnl_df, daily_pnl, initial_capital, risk_free_rate=0):
    trade_pnl = pnl_df['pnl']
    result = []
    result.append(trade_pnl.shape[0])
    result.append(trade_pnl.shape[0] / effective_year_count(daily_pnl))
    result.append(calculate_cumulative_return(daily_pnl, initial_capital))
    result.append(calculate_annual_return(daily_pnl, initial_capital))
    result.append(calculate_sharpe_ratio(daily_pnl))
    result.append(calculate_max_drawdown(daily_pnl, initial_capital))
    result.append(calculate_win_ratio(trade_pnl))
    result.append(calculate_profit_loss_ratio(trade_pnl))
    result.extend(calculate_trade_time(pnl_df))

    return pd.DataFrame(result, index=get_metrics_title()).T

def get_metrics_simple(ind_d, suffix = ''):
    return pd.DataFrame([calculate_sharpe_ratio(ind_d),
                        calculate_annual_return(ind_d,10000000),
                        calculate_max_drawdown(ind_d,10000000)], index= ['sharpe'+suffix,'annu_ret'+suffix,'mdd'+suffix]).T



# get the most important metrics
def get_key_metrics(ini_cap, pnl):
    return pd.DataFrame(
            [calculate_sharpe_ratio(pnl),
             calculate_annual_return(pnl,ini_cap),
             calculate_max_drawdown(pnl,ini_cap)],
            index = ['sharpe', 'annu_ret', 'mdd']).T


# stop loss and stop profit rules
def add_stop_lossprofit(a, t, loss_thres=0.02, profit_thres=0.02):
    refined_a = []
    if a['act'].groupby(a.reset_index().index // 2).sum().sum() != 0 or a['act'].groupby(
            a.index.date).sum().sum() != 0 or len(a) == 0:
        raise Exception('actions are not in proper format!')
        return

    for idx, ti in tqdm(pd.concat([a.reset_index().groupby(a.reset_index().index // 2).first()['index'],
                                   a.reset_index().groupby(a.reset_index().index // 2).last()['index']],
                                  axis=1).iterrows(),
                        total=a.shape[0] // 2):
        start, end = ti.iloc[0], ti.iloc[1]
        signals_in_period = t.loc[start:end][0]
        action = a.reset_index().groupby(a.reset_index().index // 2).first()['act'].iloc[idx]
        stop_profit = signals_in_period.iloc[0] - profit_thres * action
        stop_loss = signals_in_period.iloc[0] + loss_thres * action
        new_exit = signals_in_period.loc[(signals_in_period.clip(stop_loss, stop_profit) == stop_loss) | (
                    signals_in_period.clip(stop_loss, stop_profit) == stop_profit)].head(1)
        if len(new_exit) == 0:
            refined_a.append(end)
        else:
            refined_a.append(new_exit.index[0])
    return [item for pair in
            zip(a.reset_index().groupby(a.reset_index().index // 2).first()['index'].tolist(), refined_a) for item in
            pair]

def get_rolling_h(params, num_of_days):
    if type(num_of_days) == int:
        qwer = params.reset_index().groupby(params.reset_index().index//(num_of_days)).first().set_index('index')
    else:
        qwe = params.reset_index()
        qwer = qwe[qwe['index'].dt.strftime("%Y%m%d").astype(int).isin(num_of_days)].groupby(qwe['index'].dt.date).first().set_index('index')
    tmp = pd.Series(np.nan,index=params.index)
    tmp.loc[qwer.index] = qwer[0]
    h_cf = tmp.shift(-1).dropna()
    return -tmp.ffill(), pd.concat([tmp.head(1), h_cf])

def get_acf_values(stat,isDay=False):
    if not isDay:
        acf_values, confint = acf(stat.groupby([stat.index.date,stat.index.hour]).last().dropna(), nlags=500, alpha=0.05)
    else:
        acf_values, confint = acf(stat.groupby([stat.index.date]).last().dropna(), nlags=500, alpha=0.05)

    lower_bound = confint[1:, 0] - acf_values[1:]
    upper_bound = confint[1:, 1] - acf_values[1:]

    if (pd.Series(acf_values).diff().dropna()>0).sum():
        return np.argmax((acf_values[1:]>upper_bound)==0)
    else:
        return -1

def get_pacf_values(stat,isDay=False):
    if not isDay:
        pacf_values, confint = pacf(stat.groupby([stat.index.date,stat.index.hour]).last().dropna(), nlags=100, alpha=0.05)
    else:
        pacf_values, confint = pacf(stat.groupby([stat.index.date]).last().dropna(), nlags=100, alpha=0.05)

    lower_bound = confint[1:, 0] - pacf_values[1:]
    upper_bound = confint[1:, 1] - pacf_values[1:]

    if (pd.Series(pacf_values).diff().dropna()>0).sum():
        return np.argmax((pacf_values[1:]>upper_bound)==0)
    else:
        return -1

def convert(price_df):
    format_df = pd.DataFrame(price_df.values,
                        index = price_df.index.get_level_values(1))
    format_df.index = format_df.index.rename('index')
    return format_df

def cointegration_test(y, x):
    ols_result = sm.OLS(y, x).fit()
    return adfuller(ols_result.resid)

def cointegration_new_test(x,y):
    price1 = convert(get_price(x, start_date=START, end_date=END,frequency='1d')['close'])
    price2 = convert(get_price(y, start_date=START, end_date=END,frequency='1d')['close'])
    stat = (cal_regression(price1[0],price2[0])*price1[0] +price2[0])
    stat.plot()
    return adf_test(stat)

def cointegration_rolling_test(x, y):
    price1 = convert(get_price(x, start_date=START, end_date=END,frequency='1d')['close'])
    price2 = convert(get_price(y, start_date=START, end_date=END,frequency='1d')['close'])
    rolling_res = RollingOLS(price2, price1, window=30).fit()
    h_df, h_cutoff = get_rolling_h(rolling_res.params,30)
    stat= h_df*price1[0]+price2[0]
    return stat
    stat.plot()
    return adf_test(stat)

def get_k_bars(get_price_df, freq='5min'):
    result = pd.concat([get_price_df['open'].reset_index().groupby(pd.Grouper(key='datetime',freq=freq)).first()['open'],
                       get_price_df['close'].reset_index().groupby(pd.Grouper(key='datetime',freq=freq)).last()['close'],
                       get_price_df['high'].reset_index().groupby(pd.Grouper(key='datetime',freq=freq)).max()['high'],
                       get_price_df['low'].reset_index().groupby(pd.Grouper(key='datetime',freq=freq)).min()['low'],
                       get_price_df['volume'].reset_index().groupby(pd.Grouper(key='datetime',freq=freq)).sum()['volume']]
                       ,axis=1)
    result.columns = [['open','close','high','low','volume']]
    return result

def masker(my_series):
    # Find the indices where the True values start and end
    start_indices = my_series.index[my_series & ~my_series.shift(1, fill_value=False)]
    end_indices = my_series.index[my_series & ~my_series.shift(-1, fill_value=False)]

    # Create a list of intervals using the start and end indices
    intervals = [(start, end) for start, end in zip(start_indices, end_indices)]
    return intervals


# Here we will execute one timepoint after the signal
def restrain_actions(_tmp1, _tmp2, intervals, ins1=None, doubleside=True, tar=1):
    adf = pd.DataFrame()
    for i in range(len(intervals)):
        s,e = intervals[i][0],intervals[i][1]
        tmp1,tmp2 = _tmp1[s:e], _tmp2[s:e]
        concat = pd.concat([tmp1,tmp2],axis=1,keys=[0,1])
        qwe = (concat == True).astype(int).diff()
        tmp = pd.concat([concat.head(1),
                         qwe[qwe==1].dropna(how='all')],axis = 0).replace(np.nan, 0).astype(int)
        tmp[1] = -tmp[1]
        
        if not doubleside:
            if not tmp.loc[s:e].empty and (tmp.loc[s:e].iloc[0][1] == -tar or tmp.loc[s:e].iloc[0][0] == -tar):
                tmp.drop(tmp.loc[s:e].iloc[0].name, inplace=True)
        outstanding = tmp.loc[s:e].sum(axis=1).sum()
        if not tmp.loc[s:e].empty and outstanding!=0:
                if outstanding == 1:
                    res = [0, -1]
                elif outstanding == -1:
                    res = [1, 0]
                else:
                    raise Exception(f'Something wrong with outstanding {outstanding}, e {e}, tmp slice: \n {tmp.loc[s:e]}')
                tmp = pd.concat([tmp, pd.DataFrame(res, columns = [e]).T]).sort_index()
        
        adf = pd.concat([adf, tmp], axis=0)
    
    # Validation
    adf = adf.sum(axis=1)
    adf.rename('act')
    if doubleside:
        if not (adf.groupby(adf.reset_index().index//2).sum().sum() == 0):
            raise Exception('Wrong adf')
    else:
        if not ((adf.groupby(adf.reset_index().index//2).sum().sum() == 0)\
        and (adf.groupby(adf.reset_index().index//2).last().diff().sum() == 0)):
            raise Exception('Wrong adf')
    count = 0
    for i in adf.index:
        if i in np.array(intervals)[:,1].tolist():
            count += 1    
#     print(f'There are {adf.shape[0]//2} trades')
#     if adf.shape[0] != 0:
#         print(f'Hitting ratio {count / adf.shape[0] * 2}')
#     else:
#         print(f'Hitting ratio: none as 0 actions')
    
    return adf

def restrain_actions_new(tmp1, tmp2, tar=1):
    '''
    tmp1: open position
    tmp2: close position
    tar: whether to open long or short
    '''
    if tar not in (-1,1):
        raise Exception('wrong tar')
    if tar == -1:
        return -restrain_actions_new(tmp1, tmp2, 1)
    
    tmp2 = tmp2.where(tmp1.cumsum()>0,False)
    asd = (tmp1.astype(int) - tmp2.astype(int)).replace(0,np.nan).ffill().replace(np.nan,0)
    zxc = asd.diff()
    zxc.iloc[0] = asd.iloc[0]
    zxc = zxc.where((zxc == 2) | (zxc == -2) | (zxc == 1), np.nan)
    zxc = zxc.where((zxc != 1), 2) / 2
    return zxc.cumsum().ffill()


def resample_stat(stat, rolling_win, mode, freq=1, nths = 3, starting_n = -1, ffill=True):
    '''
    rolling_win: number of days in the lookback window
    freq: number of days of the sampling frequency
    '''
    stat = stat.copy()
    stat.name = None
    stat.index.name= None
    if freq == -1:
        if mode == 'mean':
            return stat.rolling(rolling_win).mean().shift(1)
        elif mode == 'std':
            return stat.rolling(rolling_win).std().shift(1)
        
    qwe = pd.Series(np.nan, index = stat.index)
    count_in_days=int(stat.dropna().groupby(stat.dropna().index.date).count().median())
    
    for i in range(nths):
        _interval = stat.reset_index().groupby(stat.reset_index().index//int(count_in_days*freq)).nth(-i+starting_n)
        if i == 0:
            interval_index = _interval['index']
            interval = _interval[0]
        else:
            interval += _interval[0]
            
    interval /= nths
    if mode == 'mean':
        qwe.loc[interval_index] = interval.rolling(int(rolling_win/freq)).mean().squeeze().values
    elif mode == 'std':
        qwe.loc[interval_index] = interval.rolling(int(rolling_win/freq)).std(ddof=0).squeeze().values
    return qwe.ffill() if ffill else qwe

def resample_stat_new(stat, win, mode,freq=1, min_periods = None):
    '''
    win: number of days in the lookback window
    freq: number of days of the sampling frequency
    '''
    if mode not in ('mean','std'):
        raise Exception('Wrong mode')
    if type(win) == str and not min_periods:
        raise Exception('Need to input a min_periods for this type of win')

    return stat.iloc[::freq].rolling(win,min_periods = min_periods).mean().reindex(stat.index).ffill() if mode == 'mean' \
      else stat.iloc[::freq].rolling(win,min_periods = min_periods).std(ddof=0).reindex(stat.index).ffill()


class bband_para():
    def __init__(self, stat,std,rolling_win_m,rolling_win_s,samp_freq_m=1,samp_freq_s=1,min_period_m=None,min_period_s=None):
        self.stat = stat
        self.stat.index.name = None
        self.stat.name = None
        self.std = std
        self.rolling_win_m = rolling_win_m
        self.samp_freq_m = samp_freq_m
        self.rolling_win_s = rolling_win_s
        self.samp_freq_s = samp_freq_s
        self.min_period_m = min_period_m
        self.min_period_s = min_period_s
    
    def get_m(self):
        return resample_stat_new(self.stat, self.rolling_win_m, 'mean', self.samp_freq_m,self.min_period_m)
    
    def get_s(self):
        return resample_stat_new(self.stat, self.rolling_win_s, 'std', self.samp_freq_s, self.min_period_s)

    
def settings_all(bband_list,open_exp='0',close_exp='0'):
    '''
    This is builder class for reversion strategy, for mom please add negative sign
    
    bband_list: the list of conditions (filters)
    open_exp & close_exp: e.g. open_exp = "0&1", which means opening position when bband_list[0]&bband_list[1]
    '''
    if type(bband_list)!=list:
        raise Exception('Wrong bband_list format!')
        
    u_upper_list,u_lower_list,l_upper_list,l_lower_list = [],[],[],[]
    stat_columns = None
    for bband in bband_list:
        if type(bband) != bband_para:
            raise Exception('Wrong bband_list input!') 
        else:
            if stat_columns is None:
                stat_columns = bband.stat.columns
            elif not bband.stat.columns.equals(stat_columns):
                raise Exception('stat should have same columns')
            stat = bband.stat
            std = bband.std
            _m = bband.get_m()
            _s = bband.get_s()
            u_upper_list.append(stat > _m + std*_s)
            l_lower_list.append(stat < _m - std*_s)
            u_lower_list.append(stat < _m)
            l_upper_list.append(stat > _m)
    
    u_upper = eval(''.join(['u_upper_list[' + char + ']' if char.isdigit() else char for char in open_exp]))
    l_lower = eval(''.join(['l_lower_list[' + char + ']' if char.isdigit() else char for char in open_exp]))
    u_lower = eval(''.join(['u_lower_list[' + char + ']' if char.isdigit() else char for char in close_exp]))
    l_upper = eval(''.join(['l_upper_list[' + char + ']' if char.isdigit() else char for char in close_exp]))
    
    g_df_u = restrain_actions_new(u_upper, u_lower, tar=1)  
    g_df_l = restrain_actions_new(l_lower, l_upper, tar=-1)  
    gdf = g_df_u.replace(np.nan,0) + g_df_l.replace(np.nan,0)
    
    if (gdf<-1).sum().sum() != 0:
        raise Exception(f"gdf doesn't conform with {(gdf<-1).sum().nlargest()}")
    if (gdf>1).sum().sum() != 0:
        raise Exception(f"gdf doesn't conform with {(gdf>1).sum().nlargest()}")
    return gdf
    
def settings(num, stat,ustat,lstat,custom_std,rolling_win,corr_intervals,samp_freq,nths,starting_n,ffill):
    _m = resample_stat(stat, rolling_win, 'mean',samp_freq, nths,starting_n,ffill)
    _s = resample_stat(stat, rolling_win, 'std',samp_freq, nths,starting_n,ffill)
    if num == 1: 
        upper_stat_outliers = (ustat > (_m + custom_std*_s))
        lower_stat_outliers = (lstat < _m) 
        t1 = upper_stat_outliers[upper_stat_outliers]
        t2 = lower_stat_outliers[lower_stat_outliers]
        a_df = restrain_actions(t1, t2, corr_intervals, stat, doubleside=False, tar=1)
        return a_df
    elif num==2:
        upper_stat_outliers = (ustat > _m) 
        lower_stat_outliers = (lstat < (_m - custom_std*_s)) 
        t1 = upper_stat_outliers[upper_stat_outliers]
        t2 = lower_stat_outliers[lower_stat_outliers]
        a_df = restrain_actions(t1, t2, corr_intervals, stat, doubleside=False, tar=-1)
        return a_df
    elif num==3 or num==4:
        upper_stat_outliers = ((ustat > (_m + custom_std*_s)) )
        lower_stat_outliers = ((lstat < (_m - custom_std*_s)) )
        t1 = upper_stat_outliers[upper_stat_outliers]
        t2 = lower_stat_outliers[lower_stat_outliers]
        a_df = restrain_actions(t1, t2, corr_intervals, stat, doubleside=True, tar=1)
        if num==3:
            return a_df
        elif num==4:
            a_df_new = pd.DataFrame()
            for i in range(a_df.shape[0]//2 - 1):
                a_df_new = pd.concat([a_df_new, 
                                      a_df.reset_index().iloc[i*2:i*2+2], 
                                      a_df.reset_index().iloc[i*2+1:i*2+3]])
            a_df_new = pd.concat([a_df_new, 
                                  a_df.reset_index().iloc[-2:]]).set_index('index')
            return a_df_new
    elif num in (-1,-2,-3,-4):
        return -settings(-num, stat,ustat,lstat,custom_std,rolling_win,corr_intervals,samp_freq,nths,starting_n,ffill)

def mixture_settings(stat,ustat,lstat,a_list, custom_std,rolling_win,corr_intervals,samp_freq=1,nths=3,starting_n=-1,toplot=False,ffill=True):
    stat,ustat,lstat = stat.copy(),ustat.copy(),lstat.copy()
    stat.name=None
    ustat.name=None
    lstat.name=None
    stat.index.name=None
    ustat.index.name=None
    lstat.index.name=None
    result = pd.DataFrame()
    for a in a_list:
        result = pd.concat([result, settings(a, stat,ustat,lstat,custom_std,rolling_win,corr_intervals,samp_freq,nths,starting_n,ffill)],axis=0)
    
    if not toplot:
        return result
    else:
        _m,_s = resample_stat(stat, rolling_win, 'mean',samp_freq, nths,starting_n,ffill), resample_stat(stat, rolling_win, 'std',samp_freq, nths,starting_n,ffill)
        return result, pd.concat([_m-custom_std*_s,stat,_m+custom_std*_s,_m],axis=1)
    
    
def settings3(num, stat,custom_std,rolling_win,shift_days, samp_freq=1,nths=1,starting_n=-1):
    _m = resample_stat(stat, rolling_win, 'mean',samp_freq, nths,starting_n)
    _s = resample_stat(stat, rolling_win, 'std',samp_freq, nths,starting_n)
    if num == -1: 
        upper_stat_outliers = (stat > (_m + custom_std*_s))
        t1 = upper_stat_outliers.loc[upper_stat_outliers.astype(int).diff()==1].astype(int)
        t2 = stat.reset_index().shift(-shift_days)
        t2.index = stat.index
        t2 = t2.loc[t1.index].set_index('index')[0]
        t2 = pd.Series(-1, index = t2.index)
        return pd.Series(list(chain.from_iterable(zip(t1, t2))),
                        index = list(chain.from_iterable(zip(t1.index, t2.index)))).to_frame()
    elif num==-2:
        lower_stat_outliers = (stat < (_m - custom_std*_s)) 
        t1 = lower_stat_outliers.loc[lower_stat_outliers.astype(int).diff()==1].astype(int)
        t2 = stat.reset_index().shift(-shift_days)
        t2.index = stat.index
        t2 = t2.loc[t1.index].set_index('index')[0]
        t2 = pd.Series(-1, index = t2.index)
        return -pd.Series(list(chain.from_iterable(zip(t1, t2))),
                        index = list(chain.from_iterable(zip(t1.index, t2.index)))).to_frame()
    
def mixture_settings3(num_list, stat, custum_std, rolling_win,shift_days):
    a_df = pd.Series()
    for num in num_list:
        a = settings3(num, stat, custum_std, rolling_win,shift_days)
        a_df = pd.concat([a_df, a], axis=0)
    return a_df

# rename column names
class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])
        
# secondary axis plotting tool
def plot2(*data):
    if len(data) == 1:
        fig = px.line(data[0])
        return fig
    elif len(data) == 2:
        data1,data2 = data
        
        # Create subplot with secondary axis
        subplot_fig = make_subplots(specs=[[{"secondary_y": True}]])

        #Put Dataframe in fig1 and fig2
        fig1 = px.line(data1)
        fig2 = px.line(data2)
        #Change the axis for fig2
        fig2.update_traces(yaxis="y2")

        #Add the figs to the subplot figure
        subplot_fig.add_traces(fig1.data + fig2.data)

        #RECOLOR so as not to have overlapping colors
        subplot_fig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
        return subplot_fig

# plot signal relevant plot
def plots(a_df, sig_df, start = None, end = None):
    '''
    a_df should not be sorted and should be only 1&-1 (meaning without position management)
    '''
    if start and type(start) == str:
        start = pd.Timestamp(start)
    if end and type(end) == str:
        end = pd.Timestamp(end)
    sell_intervals,buy_intervals=[],[]
    for i in range(int(a_df.shape[0] / 2)):
        if (start and a_df.index[2*i] > start and a_df.index[2*i+1] > start or not start) and\
            (end and a_df.index[2*i] < end and a_df.index[2*i+1] < end or not end):
            if a_df.iloc[2*i:2*i+2].iloc[0].values == 1:
                sell_intervals.append(a_df.iloc[2*i:2*i+2].index)
            if a_df.iloc[2*i:2*i+2].iloc[0].values == -1:
                buy_intervals.append(a_df.iloc[2*i:2*i+2].index)

    fig = make_subplots()
    fig1 = plot2(sig_df.loc[a_df.sort_index().loc[start:end].index])
    fig.add_traces(fig1.data)
    for i,j in buy_intervals:
        fig.add_vrect(x0=i, x1=j, line_width=0, fillcolor="green", opacity=0.2)
    for i,j in sell_intervals:
        fig.add_vrect(x0=i, x1=j, line_width=0, fillcolor="red", opacity=0.2)
    fig.show()

    
# plot strategy revelant plot
def plot(ini_cap,d,m=None,t=None,rollingsharpe=False):
    def subplot_series(tmp):
        for i in range(tmp.shape[1]):
            ind_d = tmp.iloc[:,i].dropna()
            print(f"{tmp.columns[i]}, sharpe {calculate_sharpe_ratio(ind_d):.4f}, annu_ret {calculate_annual_return(ind_d,ini_cap):.4f}, mdd {calculate_max_drawdown(ind_d,ini_cap):.4f}")
        return plot2(tmp.cumsum()/ini_cap)
    
    if type(d) == list:
        tmp = pd.concat(d,axis=1)
        tmp.rename(columns=renamer())
        return subplot_series(tmp)
    elif type(d) == pd.core.frame.DataFrame:
        tmp = d.copy()
        tmp.rename(columns=renamer())
        return subplot_series(tmp)
    else:
        if m is not None:
            display(m)
        else:
            print(f"sharpe {calculate_sharpe_ratio(d):.4f}, annu_ret {calculate_annual_return(d,ini_cap):.4f}, mdd {calculate_max_drawdown(d,ini_cap):.4f}")
        fig = make_subplots(rows=2, cols=1,specs=[[{"secondary_y": True}],[{}]], 
                            vertical_spacing=0.05,row_width=[0.2, 0.8])
        fig.add_trace(
            go.Scatter(x=d.index, y=d.cumsum()/ini_cap,name='NAV-1'),
            row=1, col=1,secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=d.index, y=calculate_drawdown(d,ini_cap),name='drawdown', opacity=0.2, fill='tozeroy',mode='none'),
            row=1, col=1,secondary_y=False
        )
        if rollingsharpe:
            fig.add_trace(
                go.Scatter(x=d.index, y=calculate_rolling_sharpe(d),name='rolling_sharpe', opacity=0.8),
                row=1, col=1,secondary_y=True
            )
        if t is not None:
            fig.add_trace(
                go.Histogram(x=t['pnl'],name='trades',nbinsx=max(100,int((t['pnl'].max()-t['pnl'].min())/t['pnl'].shape[0]/100))),
                row=2, col=1
            )

        fig.update_layout(
            height=600,
            width=950,
            margin=dict(t=20, b=20, l=50, r=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.show()

def plota(ini_cap, df, sort=None, **mapping):
    color_discrete_map= {'股指期货': 'red',
                         '国债期货': 'darkviolet',
                         '贵金属': 'gold',
                         '有色': 'orange',
                         '黑色': 'black',
                         '轻工':'cyan',
                         '石油': 'firebrick',
                         '化工': 'slategray',
                         '谷物': 'mediumspringgreen',
                         '油脂油料': 'olivedrab',
                         '软商品': 'pink',
                         '农副':'brown',
                         '航运': 'blue',
                        }
    df = df.copy()
    heading = None
    if mapping:
        for k,v in mapping.items():
            if heading is None:
                heading = pd.DataFrame(list(v.items()), columns = ['symb', k]).set_index('symb')
            else:
                heading = heading.merge(pd.DataFrame(list(v.items()), columns = ['symb', k]), 
                                        how='outer', 
                                        on='symb'
                                       ).set_index('symb').loc[df.columns]

        df = df.copy()
        col_names = ['symb']
        col_names.extend(mapping.keys())
        df.columns = pd.MultiIndex.from_arrays(heading.reset_index().values.T,
                                              names = col_names)
    else:
        df.columns = pd.MultiIndex.from_product([df.columns,[np.nan]],
                                              names = ['symb', 'nothing'])
#     print(df.columns)
    if sort:
        df.sort_index(axis=1,level=sort,ascending=False,inplace=True)

    rows,cols=math.ceil(df.shape[1] / 5),5
    fig = make_subplots(rows=rows, cols=cols,
                        shared_xaxes = 'all',
                        subplot_titles=df.columns.get_level_values('symb')
                        ,horizontal_spacing=0.05
                       )

    for i, (row, col) in enumerate(product(range(1, rows+1), range(1, cols+1))): # rows & cols are 1-indexed
        if i >= df.shape[1]:
            continue
        fig.add_trace(
            go.Scatter(x=df.index, y=df.iloc[:,i].cumsum().ffill() / ini_cap, 
                       name = df.iloc[:,i].name[0], line_color='#636EFA'),
            row=row, col=col
        )
        xaxis_name = f"xaxis{i+1}" if i != 0 else "xaxis"
        yaxis_name = f"yaxis{i+1}" if i != 0 else "yaxis"
        xaxis_showticklabels_name = f"xaxis{i+1}_showticklabels" if i != 0 else "xaxis_showticklabels"
        fig.update_layout({yaxis_name : dict(tickfont = dict(size=10)),
                           xaxis_name : dict(tickfont = dict(size=10)),
                           xaxis_showticklabels_name : True})

    if 'industry' in mapping:
        fig.for_each_trace(lambda t: t.update(line=dict(color=color_discrete_map[mapping['industry'][t.name]])))
        
    fig.update_annotations(font_size=10)
    fig.update_layout(
        height=190*rows,
        width=190*cols,
        margin=dict(t=20, b=20, l=50, r=10),
        showlegend=False
#         ,plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.show()
    
# time series apply weight
def ts_weighted_group(ini_cap, factor, pnl_matrix, cap_weight_series, underlying_pos, ranking = None, group_num = 5, samp_freq = 1):
    if ranking is None:
        ranking = pd.DataFrame()
        est = KBinsDiscretizer(n_bins=group_num, encode='ordinal', strategy='quantile')
        for i in tqdm(factor.iterrows(), total = factor.shape[0]):
            if len(i[1].dropna()) == 0:
                continue
            else:
                tmp = pd.DataFrame(est.fit_transform(i[1].dropna().values.reshape(-1,1))+1,
                                   index = i[1].dropna().index,
                                  columns = [i[0]])
                ranking = pd.concat([ranking,tmp],axis=1)
        ranking = ranking.sort_index().T
        
    result = {}
    for i in range(1,group_num+1):  
        weight_matrix = pd.DataFrame(1,index=factor.dropna(how='all').index,
                             columns = factor.dropna(how='all').columns)[ranking==i]
        weight_matrix = weight_matrix.replace(np.nan, 0)
        weight_matrix = weight_matrix*cap_weight_series
        weight_matrix = weight_matrix.div(weight_matrix.sum(1),0)
        weight_matrix = weight_matrix.iloc[::samp_freq].reindex(weight_matrix.index).ffill()
    
        weight_pnl_matrix = (pnl_matrix.loc[factor.dropna(how='all').index[0]:]*weight_matrix)
        combined_pnl = weight_pnl_matrix.sum(1)
        weight_turnover = weight_matrix.diff().abs().sum(1).mean() / 2
        pos_matrix = (weight_matrix*underlying_pos.loc[weight_matrix.index])
        turnover = pos_matrix.diff().abs().sum(1).mean() / 2
        sharpe = calculate_sharpe_ratio(combined_pnl)
        annu_ret = calculate_annual_return(combined_pnl,ini_cap)
        mdd = calculate_max_drawdown(combined_pnl,ini_cap)
        result[i] = (weight_matrix,combined_pnl,weight_turnover,turnover,sharpe,annu_ret,mdd,pos_matrix,weight_pnl_matrix)
    return ranking.copy(), result

# get ex-fee pnl
def get_netfee_pnl(combined_pnl, weight_matrix, commission, cap_weight_series, price, mul_mappings):
    netfee_pnl = combined_pnl - \
                (weight_matrix.diff().abs()*\
                 commission /\
                 cap_weight_series* \
                 price[weight_matrix.columns]*\
                 pd.DataFrame(mul_mappings.items()).set_index(0)[1].loc[weight_matrix.columns]).sum(1)
    return netfee_pnl.loc[combined_pnl.index[0]:combined_pnl.index[-1]]

# sensitivity tools
def sensitivity(func, *args, tqdm_use=True, **kwargs):
    '''
    requires func to take the input of *args and **kwargs and output of a single row dataframe with proper column names
    '''
    result = pd.DataFrame()
    list_dict = {}
    for k,v in kwargs.items():
        list_dict[k] = v
    result_index = pd.MultiIndex.from_product(list(list_dict.values()), names = list(list_dict.keys()))
    
    # just a format change, the 2 blocks are identical
    if tqdm_use:
        for i in tqdm(range(len(result_index))):
            try:
                result = pd.concat([result,func(*args,**dict(zip(result_index.names,result_index[i])))],axis=0)
            except:
                res = pd.DataFrame([np.nan])
                result = pd.concat([result,res],axis=0)
    else:
        for i in range(len(result_index)):
            print(f"{i}/{len(result_index)}")
            try:
                result = pd.concat([result,func(*args,**dict(zip(result_index.names,result_index[i])))],axis=0)
            except:
                res = pd.DataFrame([np.nan])
                result = pd.concat([result,res],axis=0)
    
    if 0 in result.columns:
        result.drop([0],axis=1)
    result.index = result_index
    return result

# determine secondary contract
def get_dom2(symb):
    s = symb[:-4]
    y = int(symb[-4:-2])
    m = symb[-2:]
    res = []
    for sche in ROLLOVER_SCHEDULE[s]:
        y_add = 1 if int(sche[m])<int(m) else 0
        res.append(s+str(y+y_add)+sche[m])
        m = sche[m]
        y += y_add
    return res
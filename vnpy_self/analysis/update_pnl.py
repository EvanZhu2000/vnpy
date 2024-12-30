import akshare as ak
from datetime import datetime
import pandas as pd
import sys

def run(date_str:str):
    pnl_directory = '//192.168.91.128/share_folder/Evan/PNL.csv'
    tmp =  pd.read_excel(f'G:\My Drive\\2520000355逐笔_{date_str}.xls')
    
    # ========= 资金状况 =============
    settlement_total_df = pd.concat([
        tmp.iloc[11:25][['Unnamed: 0', 'Unnamed: 3']].rename(columns={'Unnamed: 0':'fields','Unnamed: 3':'values'}),
        tmp.iloc[11:25][['Unnamed: 6', 'Unnamed: 9']].rename(columns={'Unnamed: 6':'fields','Unnamed: 9':'values'})
    ]).dropna().set_index('fields')
    settlement_total_df.loc['风 险 度 Risk Degree:','values'] = float(settlement_total_df.loc['风 险 度 Risk Degree:','values'].strip('%').strip())/100
    settlement_total_df = settlement_total_df.astype(float).round(4)

    # ========= 成交记录 =============
    try:
        settlement_trades_df = tmp.iloc[tmp.loc[tmp['Unnamed: 0'] == '成交记录'].index[0]:tmp.loc[(tmp['Unnamed: 0'].str.contains('共'))&(tmp['Unnamed: 0'].str.contains('条'))].index[0]].dropna()
        settlement_trades_df.columns = settlement_trades_df.iloc[0]
        settlement_trades_df = settlement_trades_df.iloc[1:].reset_index(drop=True)
    except:
        settlement_trades_df = pd.DataFrame()


    cur_bal = settlement_total_df.to_dict()['values']['客户权益 Client Equity:']
    
    fx_spot_quote_df = ak.fx_spot_quote()
    hkdcny = fx_spot_quote_df.loc[fx_spot_quote_df['货币对'] == 'HKD/CNY'].eval('(买报价+卖报价)/2').values[0]
    
    ### read csv
    records = pd.read_csv(pnl_directory)
    last_day_records = records.sort_values('Date', ascending=True).iloc[-1]
    pre_bal, cum_pnl_percentage = last_day_records['Balance'], last_day_records['CUM_PNL%']
    
    cur_bal = round(cur_bal,0)
    pnl_CNY = round(cur_bal - pre_bal,2)
    pnl_HKD = round(pnl_CNY / hkdcny,2)
    pnl_percentage = round((pnl_CNY / pre_bal), 4)
    cum_pnl_percentage += round(pnl_percentage, 4)
    
    records.loc[len(records)] = [date_str, cur_bal, pnl_CNY, pnl_HKD, pnl_percentage, cum_pnl_percentage]
    records.set_index('Date', inplace=True)

    ### write csv
    records.to_csv(pnl_directory)
    print('all finished')


if __name__ == "__main__":
    # date_str = datetime.today().strftime("%Y%m%d")
    date_str = '20241227'
    run(date_str)

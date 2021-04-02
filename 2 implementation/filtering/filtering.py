import numpy as np
import pandas as pd

mth_symbol = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(level='SYMBOL')]
mth_ind = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(key='IND_CODE')]
mth = [pd.Grouper(freq='M', level='DATE'), ]
symbol = [pd.Grouper(level='SYMBOL'), ]

mthly = pd.read_csv('../../_data/data.csv')
mthly = mthly.dropna(subset=['SYMBOL', 'PCT_RET', 'VOLUME'])
mthly = mthly.set_index(['DATE', 'SYMBOL']).sort_index()
stocks = mthly.index.get_level_values('SYMBOL').drop_duplicates()
dates = mthly.index.get_level_values('DATE').drop_duplicates()
mthly = mthly.reindex(pd.MultiIndex.from_product((dates, stocks), names=('DATE', 'SYMBOL')), fill_value=np.nan).sort_index()
mthly.index = mthly.index.set_levels([
        pd.to_datetime(mthly.index.levels[0]),
        mthly.index.levels[1],
        ])

daily = pd.read_csv('../../_data/daily_prices.csv', usecols=['SYMBOL', 'DATE', 'PCT_RET', 'VOLUME', 'CLOSE'])
daily['DATE'] = pd.to_datetime(daily['DATE'])
daily = daily.set_index(['DATE', 'SYMBOL'])

mthly['ADV'] = (daily['CLOSE'] * daily['VOLUME']).groupby(mth_symbol).mean()
mthly = mthly[(mthly['HSI'] == True) | (mthly['HSCI'] == True) | (mthly['HSCEI'] == True)]

df = mthly[['ADV', 'MKT_CAP']]

#a = df[
#   (df['ADV'] > 4e7) & \
#   (df['MKT_CAP'] > 7.78e3)
#   ]

#adv_mask = df.index.isin(df['ADV'].groupby(mth, group_keys=False).apply(lambda x: x.sort_values()[-int(x.count() * 0.40):]).index)
adv_mask = df['ADV'] > 4e7
mktcap_mask = df['MKT_CAP'] > 7.78e3
a = df[adv_mask & mktcap_mask]

b = a['ADV'].groupby(mth).count()
b.plot(figsize=(12,6), grid=True)

b.rolling(12).sum().plot(figsize=(12,6), grid=True)
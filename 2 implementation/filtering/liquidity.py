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

mthly_comp = mthly[(mthly['HSI'] == True) | (mthly['HSCI'] == True) | (mthly['HSCEI'] == True)]

HSI = mthly[(mthly['HSI'] == True)]
HSCI = mthly[(mthly['HSCI'] == True)]
HSCEI = mthly[(mthly['HSCEI'] == True)]

HSI['PCT_RET'].groupby(mth).count().plot()
HSCI['PCT_RET'].groupby(mth).count().plot()
HSCEI['PCT_RET'].groupby(mth).count().plot()

daily = pd.read_csv('../../_data/daily_prices.csv', usecols=['SYMBOL', 'DATE', 'PCT_RET', 'VOLUME', 'CLOSE'])
daily['DATE'] = pd.to_datetime(daily['DATE'])
daily = daily.set_index(['DATE', 'SYMBOL'])

from pandas.tseries.offsets import MonthEnd
daily['DATE_MTHEND'] = pd.to_datetime(daily.index.get_level_values('DATE'), format="%Y%m") + MonthEnd(0)
daily = daily.reset_index(drop=False)
daily.columns = ['DATE_REAL', 'SYMBOL', 'CLOSE', 'PCT_RET', 'VOLUME', 'DATE']
daily = daily.set_index(['DATE', 'SYMBOL'])

daily = daily[daily.index.isin(mthly_comp.index)]

daily = daily.reset_index(drop=False)
daily.columns = ['DATE_MTHEND', 'SYMBOL', 'DATE', 'CLOSE', 'PCT_RET', 'VOLUME']
daily = daily.set_index(['DATE', 'SYMBOL'])


a = (daily['VOLUME'] * daily['CLOSE']).groupby(mth_symbol).mean()
d = a.groupby(mth).count()
d.plot(figsize=(12,6), grid=True)
b = a[a > 40000000]
c = b.groupby(level='DATE').count()
c.plot(figsize=(12,6), grid=True)

(c/d).plot(figsize=(12,6), grid=True)

ax = a.dropna().groupby(level='DATE').apply(lambda x: x.sort_values()[int(x.count() * (1 - 0.35))]).plot(figsize=(12,6), grid=True)
ax.ticklabel_format(style='plain', axis='y')
ax.set_ylim([0,200000000])

ax = a.dropna().groupby(level='DATE').apply(lambda x: x.sort_values()[-200]).plot(figsize=(12,6), grid=True)
ax.ticklabel_format(style='plain', axis='y')

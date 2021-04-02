import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


i = 94


returns_path = 'returns{}.csv'.format(i)
result_path = 'results{}.csv'.format(i)

df = pd.read_csv(returns_path)
df = df.dropna()
df = df.set_index('DATE')

mkt = 'HSI_PCT_RET'
rf = 'RF'
cols_not_included = ['HSI_PRE', 'HSI_OPEN', 'HSI_HIGH', 'HSI_LOW', 'HSI_CLOSE', 'HSI_VOLUME', 'HSI_PRICE_CHG', 'HSCI_PRE', 'HSCI_OPEN', 'HSCI_HIGH', 'HSCI_LOW', 'HSCI_CLOSE', 'HSCI_PRICE_CHG', 'HSCEI_PRE', 'HSCEI_OPEN', 'HSCEI_HIGH', 'HSCEI_LOW', 'HSCEI_CLOSE', 'HSCEI_VOLUME', 'HSCEI_PRICE_CHG', 'RF']
cols = df.columns[~df.columns.isin(cols_not_included)]


results = pd.DataFrame()

for col in cols:

    # For CAPM model, y = portfolio risk-premium, x = market risk-premium
    series = df[col]
    series_mkt = df[mkt]
    series_rf = df[rf]        
    y = series - series_rf
    x = series_mkt - series_rf
    series_1base = series / 100 + 1
    series_cum = series_1base.cumprod()
    win_market = series > series_mkt
    
    # CAPM model
    model = sm.OLS(y, sm.add_constant(x)).fit()
    
    metrics = {
            'mth_ret_avg': series.mean(),
            'compound_ret': (stats.gmean(series_1base) - 1) * 100,
            'mth_ret_std': series.std(),        
            'alpha': model.params['const'],
            'beta': model.params[0],
            'r-squared': model.rsquared,
            'sharpe ratio': y.mean() / y.std(),
            'max_%_drawdown': ((series_cum / series_cum.cummax() - 1)*100).min(),
            'mth_ret_max': series.max(),
            'mth_ret_min': series.min(),
            '% above market': win_market.sum() / win_market.count(),
            'corr with index': series.corr(series_mkt),
            }
    
    index = metrics.keys()
    values = metrics.values()
    result = pd.DataFrame(data=values, index=index, columns=[col])
    
    results = pd.concat([results, result], axis=1)
    
results.to_csv(result_path)


#a = [6000, 1030000, 2000000, 50000, 60000, 110000, 48000, 510000, 430000, 400000, 218000, 52000, 76000]
#b = [980, 240000, 338000, 9800, 14500, 19300, 19650, 98600, 66000, 63000, 54000, 20000, 26000]


#a = df[['L10_stopNone', 'HSI_PCT_RET']]
#a.index = pd.to_datetime(a.index).strftime("%Y")
#ax = a.plot(grid=True, style='.', figsize=(12, 6), rot=45, title='Monthly Return of HSI vs Long-10')
#ax.legend(["Long 10", "HSI"])
#ax.set_ylabel("Monthly Return (%)")
#ax.set_xlabel("Date")
#
#ax = a.rolling(12).mean().plot(grid=True, style='.', figsize=(12, 6), rot=45, title='Monthly Return of HSI vs Long-10 (MA 12 months)')
#ax.legend(["Long 10", "HSI"])
#ax.set_ylabel("Monthly Return (%)")
#ax.set_xlabel("Date")
#
#
#a = df[['L10_stop5', 'HSI_PCT_RET']]
#a.index = pd.to_datetime(a.index).strftime("%Y")
#ax = a.plot(grid=True, style='.', figsize=(12, 6), rot=45, title='Monthly Return of HSI vs Long-10 with 5% stopping')
#ax.legend(["Long 10", "HSI"])
#ax.set_ylabel("Monthly Return (%)")
#ax.set_xlabel("Date")
#
#ax = a.rolling(12).mean().plot(grid=True, style='.', figsize=(12, 6), rot=45, title='Monthly Return of HSI vs Long-10 with 5% stopping (MA 12 months)')
#ax.legend(["Long 10", "HSI"])
#ax.set_ylabel("Monthly Return (%)")
#ax.set_xlabel("Date")



#a = df[['L10_stopNone', 'HSI_PCT_RET']]
#a.index = pd.to_datetime(a.index)
#a = a / 100 + 1
#a = a.groupby([pd.Grouper(freq='Y', level='DATE'), ]).apply(lambda x: np.prod(x))
#a = (a ** (1/12) - 1) * 100
#a.index = a.index.strftime("%Y")
#
#ax = a.plot(grid=True, style='.', figsize=(12, 6), rot=45, title='Yearly Average of HSI vs Long-10')
#ax.legend(["Long 10", "HSI"])
#ax.set_ylabel("Monthly Return (%)")
#ax.set_xlabel("Date")
#
#
#a = df[['L10_stop5', 'HSI_PCT_RET']]
#a.index = pd.to_datetime(a.index)
#a = a / 100 + 1
#a = a.groupby([pd.Grouper(freq='Y', level='DATE'), ]).apply(lambda x: np.prod(x))
#a = (a ** (1/12) - 1) * 100
#a.index = a.index.strftime("%Y")
#
#ax = a.plot(grid=True, style='.', figsize=(12, 6), rot=45, title='Yearly Average of HSI vs Long-10 with 5% stopping')
#ax.legend(["Long 10", "HSI"])
#ax.set_ylabel("Monthly Return (%)")
#ax.set_xlabel("Date")








#import matplotlib.pyplot as plt
#plt.figure(figsize=(8,8))
#a = df[['HSI_PCT_RET', 'L10_stopNone']]
#ax = a.plot.scatter('HSI_PCT_RET', 'L10_stopNone', ax = plt.gca())
#ax.plot((-20, 20), (-20, 20), 'r-')
#ax.set_ylabel("Long 10")
#ax.set_xlabel("HSI")
#
#plt.figure(figsize=(8,8))
#a = df[['HSI_PCT_RET', 'L10_stop5']]
#ax = a.plot.scatter('HSI_PCT_RET', 'L10_stop5', ax = plt.gca())
#ax.plot((-20, 20), (-20, 20), 'r-')
#ax.set_ylabel("Long 10 with 5% stopping")
#ax.set_xlabel("HSI")






#model_infos = np.load('../_data/model_info{}.npy'.format(i))
#model_infos = model_infos[()]
#
#
#date = '2018-12-31'
#model_info = model_infos[date]
#coef = model_info['coef']
#var = model_info['valid variables']
#
#data = pd.DataFrame({
#        'coef': coef,
#        'var': var,
#        }).set_index('var')
#data = data[abs(data) > 5e-5].dropna()
#data['mag'] = abs(data['coef'])
#data['sign'] = -np.sign(data['coef'])
#data = data.sort_values('mag', ascending=0)
#data = data[['mag', 'sign']]
#d = data.head(10)
#d.to_clipboard()
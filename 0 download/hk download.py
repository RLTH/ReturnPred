import pandas as pd
from util import txt_to_list



import pdblp
con = pdblp.BCon(debug=True, timeout=5000)
con.start()

indexes = ['HSI', 'HSCI', 'HSCEI']
stocks = txt_to_list('SYMBOLS.txt')

################################################################################ Dates

years = range(2003, 2019)
months = range(1, 13)
dates = []
for year in years:
    for month in months:
        dates.append('{:04d}-{:02d}-01'.format(year, month))
dates = pd.to_datetime(pd.Index(data=dates)).shift(periods=1, freq='M')

################################################################################ Historical index components

#for index in indexes:
#    df = con.bulkref_hist(
#            tickers = '{} Index'.format(index),
#            flds = 'INDX_MWEIGHT_HIST',
#            dates = dates.strftime('%Y%m%d').values,
#            date_field = "END_DATE_OVERRIDE"
#            )
#    
#    df.to_csv('../_data/_index/{}_histcom.csv'.format(index), index = False)
    
################################################################################ Symbols
    
#def read_index_histcom(path):
#    df = pd.read_csv(path)
#    df = df[['date', 'name', 'value', 'position']]
#    df = df.groupby('date').apply(lambda x: x.pivot(index='position', columns='name', values='value'))
#    df = df.reset_index(drop=False)
#    df.columns = ['DATE', 'POSITION', 'SYMBOL', 'WEIGHT']
#    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
#    df['SYMBOL'] = df['SYMBOL'] + ' Equity'
#    df = df.set_index(['DATE', 'SYMBOL'])
#    return df
#
#hsi_df = read_index_histcom('../_data/_index/hsi_histcom.csv')
#hsci_df = read_index_histcom('../_data/_index/hsci_histcom.csv')
#hscei_df = read_index_histcom('../_data/_index/hscei_histcom.csv')
#
#hsi = hsi_df.index.get_level_values('SYMBOL').unique()
#hsci = hsci_df.index.get_level_values('SYMBOL').unique()
#hscei = hscei_df.index.get_level_values('SYMBOL').unique()
#
#all_ind = hsi.union(hsci)
#all_ind = all_ind.union(hscei)
#
#with open('SYMBOLS.txt', 'w') as f:
#    f.write('\n'.join(list(all_ind)))

################################################################################ Index monthly

#for index in indexes:
#    flieds = txt_to_list('flds_index.txt')
#    df = con.bdh(
#            tickers = '{} Index'.format(index),
#            flds = flieds,
#            start_date = '20030101',
#            end_date = '20181231',
#            elms = [
#                    ("periodicityAdjustment", "ACTUAL"),
#                    ("periodicitySelection", "MONTHLY"),
#                    ("nonTradingDayFillOption", "ALL_CALENDAR_DAYS"), # "ACTIVE_DAYS_ONLY"
#                    ("nonTradingDayFillMethod", "NIL_VALUE"), # "PREVIOUS_VALUE"
#                    ("adjustmentNormal", True),
#                    ("adjustmentAbnormal", True),
#                    ("adjustmentSplit", True),
#                    ]
#            )
#    
#    df.to_csv('../_data/_index/{}.csv'.format(index))

################################################################################ riskfree rate monthly


# If you go to Bloomberg, FLDS, then check a Hong Kong stock, say "1 HK Equity", then input "risk free" as the query.
# You can see that the risk free rate for HK is the "HKGG10Y Index"


#rf = con.bdh(
#        tickers = 'HKGG10Y Index',
#        flds = 'PX_LAST',
#        start_date = '20030101',
#        end_date = '20181231',  
#        elms = [
#                ("periodicityAdjustment", "ACTUAL"),
#                ("periodicitySelection", "MONTHLY"),
#                ("nonTradingDayFillOption", "ALL_CALENDAR_DAYS"), # "ACTIVE_DAYS_ONLY"
#                ("nonTradingDayFillMethod", "NIL_VALUE"), # "PREVIOUS_VALUE"
#                ("adjustmentNormal", True),
#                ("adjustmentAbnormal", True),
#                ("adjustmentSplit", True),
#                ],
#        longdata = True
#        )
#
#rf.to_csv('../_data/_rf/bdh.csv', index = False)




################################################################################ flds monthly

#for i in [1, 2, 3, 4, 5, 6, 7, 8]:
#    flieds = txt_to_list('flds{}.txt'.format(i))
#    
#    df1 = con.bdh(
#            tickers = stocks, 
#            flds = flieds,
#            start_date = '20030101',
#            end_date = '20181231',  
#            elms = [
#                    ("periodicityAdjustment", "ACTUAL"),
#                    ("periodicitySelection", "MONTHLY"),
#                    ("nonTradingDayFillOption", "ALL_CALENDAR_DAYS"), # "ACTIVE_DAYS_ONLY"
#                    ("nonTradingDayFillMethod", "NIL_VALUE"), # "PREVIOUS_VALUE"
#                    ("adjustmentNormal", True),
#                    ("adjustmentAbnormal", True),
#                    ("adjustmentSplit", True),
#                    ],
#            longdata = True
#            )
#    
#    df1.to_csv('../_data/{}/bdh.csv'.format(i), index = False)


################################################################################ flds monthly with ref

#q = len(stocks) / 4
#i = 4
#
#flieds_ref = txt_to_list('ref.txt')
#
#df2 = con.ref_hist(
#        tickers = stocks[int(i * q) : int((i + 1) * q)], 
#        flds = flieds_ref,
#        dates = dates
#        )
#
#df2.to_csv('../_data/_ref/ref{}.csv'.format(i), index = False)


################################################################################ flds daily

#flieds = txt_to_list('flds1.txt')
#
#df1 = con.bdh(
#        tickers = stocks, 
#        flds = flieds,
#        start_date = '20030101',
#        end_date = '20181231',  
#        elms = [
#                ("periodicityAdjustment", "ACTUAL"),
#                ("periodicitySelection", "DAILY"),
#                ("nonTradingDayFillOption", "ALL_CALENDAR_DAYS"), # "ACTIVE_DAYS_ONLY"
#                ("nonTradingDayFillMethod", "NIL_VALUE"), # "PREVIOUS_VALUE"
#                ("adjustmentNormal", True),
#                ("adjustmentAbnormal", True),
#                ("adjustmentSplit", True),
#                ],
#        longdata = True
#        )
#
#df1.to_csv('../_data/_daily/bdh.csv', index = False)

################################################################################ est

#for est_yr in [0, 1, 2]:
#    for part in [1, 2]:
#        flieds = txt_to_list('est{}.txt'.format(part))
#        
#        est_df = con.bdh(
#                tickers = stocks, 
#                flds = flieds,
#                start_date = '20030101',
#                end_date = '20181231',  
#                elms = [
#                        ("periodicityAdjustment", "ACTUAL"),
#                        ("periodicitySelection", "MONTHLY"),
#                        ("nonTradingDayFillOption", "ALL_CALENDAR_DAYS"), # "ACTIVE_DAYS_ONLY"
#                        ("nonTradingDayFillMethod", "NIL_VALUE"), # "PREVIOUS_VALUE"
#                        ("adjustmentNormal", True),
#                        ("adjustmentAbnormal", True),
#                        ("adjustmentSplit", True),
#                        ],
#                ovrds = [
#                        ('BEST_FPERIOD_OVERRIDE', '{}FY'.format(est_yr)),
#                        ],
#                longdata = True
#                )
#        
#        est_df.to_csv('../_data/_est/fy{}_{}.csv'.format(est_yr, part), index = False)


import pandas as pd
import numpy as np

################################################################################

lag_period = 12
rank_x = True
rank_y = True
binary_y_return_sign = False
binary_y_topbot = False
topbotcount = False


def data_formatting(raw_df):
    # Putting back the missing rows from (dates * stocks)
    raw_df = raw_df.dropna(subset=['SYMBOL', 'PCT_RET']) # , 'VOLUME'
    raw_df = raw_df.set_index(['DATE', 'SYMBOL']).sort_index()
    stocks = raw_df.index.get_level_values('SYMBOL').unique()
    dates = raw_df.index.get_level_values('DATE').unique()
    raw_df = raw_df.reindex(pd.MultiIndex.from_product((dates, stocks), names=('DATE', 'SYMBOL')), fill_value=np.nan).sort_index()
    
    
    raw_df.index = raw_df.index.set_levels([
            pd.to_datetime(raw_df.index.levels[0]),
            raw_df.index.levels[1],
            ])

    return raw_df

def chunk_reading(path):
    data = pd.read_csv(path, sep = ',', engine = 'python', iterator = True)
    loop = True
    chunkSize = 10000
    chunks = []
    index = 0
    while loop:
        try:
            print(index)
            chunk = data.get_chunk(chunkSize)
            chunks.append(chunk)
            index += 1
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    data = pd.concat(chunks, ignore_index = True)
    return data


#######################################################################
#                            Data Loading                             #
#######################################################################

def loading(x_var_selection, ADV_filter):
    monthly = pd.read_csv('../../_data/data.csv')
    daily = pd.read_csv('../../_data/daily.csv')
    
#    monthly = chunk_reading('../../_data/data.csv')
#    daily = chunk_reading('../../_data/daily.csv')
    
    monthly = data_formatting(monthly)
    daily = data_formatting(daily)
    
    print('data loaded')
    
    lag_period = 12
    dates = monthly.index.get_level_values('DATE').drop_duplicates()
    valid_dates = dates[lag_period:]
    
    # One hot encoding for industry
    industry_column_name = 'IND_CODE'
    if industry_column_name in x_var_selection:
        dummies = pd.get_dummies(monthly, columns=[industry_column_name])
        monthly = pd.concat([dummies, monthly['IND_CODE']], axis=1) 
        allvariables = list(monthly.columns)
        industryvariable = [x for x in allvariables if industry_column_name in x and industry_column_name != x]
        x_var_selection = x_var_selection + industryvariable
        x_var_selection.remove(industry_column_name)
    

    
    # suspended stocks are filtered out
    monthly['invalid'] = ( monthly['VOLUME'].isna() | monthly['PCT_RET'].isna() ).groupby(level='SYMBOL', group_keys=False).rolling(3).sum()
    monthly['invalid'] = monthly['invalid'].astype(bool)    
    monthly = monthly[~monthly['invalid']]
    
    
    # remove the first lag_period months
    monthly = monthly[monthly.index.get_level_values('DATE').isin(valid_dates)]
    
    
    
    # correcting survivial bias
    monthly = monthly[(monthly['HSI'] == True) | (monthly['HSCI'] == True) | (monthly['HSCEI'] == True)] #%
    
    
    
    # ADV and MKT_CAP filtering
    if ADV_filter:
        adv_threshold = 4e7
        mktcap_threshold = 7.78e3
        
        mth_symbol = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(level='SYMBOL')]
        
        adv = (daily['CLOSE'] * daily['VOLUME']).groupby(mth_symbol).mean()
        mktcap = monthly['MKT_CAP']
        adv_date_mask = adv.index.get_level_values('DATE') <= '2006-12-31'
        mktcap_date_mask = mktcap.index.get_level_values('DATE') <= '2006-12-31'
        
        # The thresholds are halved before 2007 to allow for larger sample size
        adv_mask = adv > ( adv_date_mask * (adv_threshold / 2) + (~adv_date_mask) * adv_threshold )
        mktcap_mask = mktcap > ( mktcap_date_mask * (mktcap_threshold / 2) + (~mktcap_date_mask) * mktcap_threshold )
                
#        adv_mask.index = adv_mask.index.set_levels([
#                adv_mask.index.levels[0].strftime('%Y-%m-%d').astype(object),
#                adv_mask.index.levels[1],
#                ])
        
        monthly = monthly[adv_mask & mktcap_mask]
    
    
    # filling missing value with mean from valid stocks
    monthly = monthly.groupby(level='DATE', group_keys=False).transform(lambda x: x.fillna(x.mean()))
        
        
        

    return monthly, daily, x_var_selection


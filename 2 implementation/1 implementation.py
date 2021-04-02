import pandas as pd
import numpy as np
import json


#from statistics import mean
from scipy import stats


import warnings
warnings.simplefilter(action='ignore')

################################################################################


warmstart = False
#np.random.seed(seed=0)



LS_counts = [10] #[10, 20, 30]
corr_thresholds = [0.2] #[0, 0.1, 0.2] #[-0.1, 0, 0.1, 0.2, 0.3]
ret_thresholds = [1.5] #[1.5, 2, 2.5] #[0.5, 1, 1.5, 2, 2.5]
stop_thresholds = [None, 5] #[None, 2, 3, 5, 7, 9, 10, 11]

#LS_counts = [10, 20, 30]
#corr_thresholds = [-0.1, 0, 0.1, 0.2, 0.3]
#ret_thresholds = [0.5, 1, 1.5, 2, 2.5]
#stop_thresholds = [None, 2, 3, 5, 7, 9, 10, 11]

#LS_counts = [10] 
#corr_thresholds = [0, 0.1, 0.2]
#ret_thresholds = [1.5, 2, 2.5]
#stop_thresholds = [None]


#    ['L10', 'L20', 'L30',\
#      'LS10', 'LS20', 'LS30',\
#      'L10_stop', 'L20_stop', 'L30_stop',\
#      'LS10_stop', 'LS20_stop', 'LS30_stop',\
#      'LSc10', 'LSc20', 'LSc30']
#      'ind_L5', 'ind_L10', 'ind_Lall',\
#      'ind_LS5', 'ind_LS10', 'ind_LSall']

#################################################################################

from strategy import invest


i = 94



if not warmstart:

    #######################################################################
    #                            Data Loading                             #
    #######################################################################

    pred_df = pd.read_csv('../_data/pred{}.csv'.format(i))
    pred_df = pred_df.set_index(['DATE', 'SYMBOL'])
    
#    pred_df = pred_df.dropna(subset=['pred_y'], how='all', axis=0) #%
    
    daily = pd.read_csv('../_data/daily.csv')
    daily['DATE'] = pd.to_datetime(daily['DATE'])
    daily = daily.set_index(['DATE', 'SYMBOL'])

    print('data loaded')
    
    hist_corr = np.load('../_data/hist_corr.npy')
    
    print('np data loaded')
    

    
dates = pred_df.index.get_level_values('DATE').drop_duplicates()
returns = pd.read_csv('../_data/mkt.csv', index_col=0)

#######################################################################
#                               Main Loop                             #
#######################################################################

for date, pred_data in pred_df.groupby(level='DATE'):

#    if date != '2018-12-31':
#        continue
#    else:
#        import sys
#        sys.exit()
##        pass
    
    pred_data = pred_data.droplevel('DATE')
    corr = hist_corr.item().get(date)
    
    daily_xs = daily['PCT_RET'].unstack()[daily.unstack().index.strftime('%Y-%m') == pd.to_datetime(date).strftime('%Y-%m')]

    for LS_count in LS_counts:
        for corr_threshold in corr_thresholds:
            for ret_threshold in ret_thresholds:
                for stop_threshold in stop_thresholds:
                    print(date, corr_threshold, ret_threshold, stop_threshold)

                    arguments = {
                            'pred_data': pred_data,
                            'LS_count': LS_count,
                            }

                    returns.loc[date, 'EQWEIGHT_in_test_pool'] = invest('eq', **arguments) 
                    
                    arguments = {
                            'pred_data': pred_data,
                            'LS_count': LS_count,
                            'daily_xs': daily_xs,
                            'stop_threshold': stop_threshold,
#                            'stop_method': 'individual',
                            }
                    
                    returns.loc[date, 'L{}_stop{}'.format(LS_count, stop_threshold)] = invest('L', **arguments)
                    returns.loc[date, 'S{}_stop{}'.format(LS_count, stop_threshold)] = invest('S', **arguments)
                    returns.loc[date, 'LS{}_stop{}'.format(LS_count, stop_threshold)] = invest('LS', **arguments)
                    
#                    returns.loc[date, 'S{}_stop{}'.format(LS_count, stop_threshold)], \
#                    returns.loc[date, 'count'], \
#                    returns.loc[date, 'not_stopped'], \
#                    returns.loc[date, 'recovered'], \
#                    returns.loc[date, 'not_recovered'], \
#                    returns.loc[date, 'not_stopped_return'], \
#                    returns.loc[date, 'loss_from_recovered_return'], \
#                    returns.loc[date, 'gain_from_not_recovered_return'], \
#                    returns.loc[date, 'realized_recovered_return'], \
#                    returns.loc[date, 'realized_not_recovered_return'], \
#                     = invest('S', **arguments)
                    
                    arguments = {
                            'pred_data': pred_data,
                            'LS_count': LS_count,
                            'daily_xs': daily_xs,
                            'corr': corr,
                            'corr_threshold': corr_threshold,
                            'ret_threshold': ret_threshold,
                            'stop_threshold': stop_threshold,
#                            'stop_method': 'individual',
                            }
                    
                    
                    returns.loc[date, 'LSc3{}_{}_{}_stop{}'.format(LS_count, corr_threshold, ret_threshold, stop_threshold)] = invest('LSc', **arguments)
                    returns.loc[date, 'LScp2{}_{}_{}_stop{}'.format(LS_count, corr_threshold, ret_threshold, stop_threshold)] = invest('LScp', **arguments)
            


returns.to_csv('returns{}.csv'.format(i))

print('done {}'.format(i))
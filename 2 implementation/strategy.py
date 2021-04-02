import numpy as np
import pandas as pd


# strategies = ['L', 'S', 'LS', 'LSc', 'LScp']
# stop_methods = ['profolio', 'individual']

def invest(strategy, pred_data, LS_count, daily_xs=None, corr=None, corr_threshold=None, ret_threshold=None, stop_threshold=None, stop_method='profolio'):
    
    if ret_threshold is not None:
        pred_data['score'] = ( pred_data['pred_y'] - pred_data['pred_y'].mean() ) / pred_data['pred_y'].std()
        pred_data = pred_data[ (pred_data['score'] >= ret_threshold) | (pred_data['score'] <= -ret_threshold) ]
    
    if strategy in ['L', 'S', 'LS']:
        L_count, S_count = LS_count, LS_count
        if strategy == 'L':
            S_count = 0
        if strategy == 'S':
            L_count = 0
        tops = pred_data.sort_values('pred_y_rank').tail(L_count).index
        bots = pred_data.sort_values('pred_y_rank').head(S_count).index
    elif strategy in ['eq']:
        tops = pred_data.index
        bots = pd.Index([])
    elif strategy in ['LSc']:
        init_tops = pred_data.sort_values('pred_y_rank', ascending = False).index
        tops = pd.Index([])
        bots = pd.Index([])
        for top in init_tops:
            stock_corr = corr[top]
            most_corr_stocks = stock_corr.loc[(stock_corr >= corr_threshold) & (stock_corr.index != top)].index
            bot_stocks_df = pred_data[pred_data.index.isin(most_corr_stocks)].sort_values('pred_y_rank')            
            bot = bot_stocks_df.head(1).index
            
            
            if tops.size < LS_count:
                tops = tops.append(pd.Index([top]))
            if bots.size < LS_count and not bot.empty:
                bots = bots.append(bot)
                
            if tops.size >= LS_count and bots.size >= LS_count:
                break
    elif strategy in ['LScp']:
        all_pairs_corr = corr.stack()
        most_corr_pairs = all_pairs_corr[(all_pairs_corr >= corr_threshold) & (all_pairs_corr != 1)].to_frame()
        most_corr_pairs = most_corr_pairs[most_corr_pairs.index.get_level_values(0).isin(pred_data.index) &\
                                          most_corr_pairs.index.get_level_values(1).isin(pred_data.index)]
        
        Long_stocks_return = pred_data.loc[most_corr_pairs.index.get_level_values(0), 'pred_y'].values
        Short_stocks_return = pred_data.loc[most_corr_pairs.index.get_level_values(1), 'pred_y'].values
        most_corr_pairs['pred_y'] = Long_stocks_return - Short_stocks_return
        most_corr_pairs = most_corr_pairs[most_corr_pairs['pred_y'] >= 0]
        
        top_pairs = most_corr_pairs.sort_values('pred_y').tail(LS_count)
        tops = top_pairs.index.get_level_values(0)
        bots = top_pairs.index.get_level_values(1)
        
    
    import json
    portfolio = json.dumps({'L': list(tops), 'S': list(bots)})
    
    if len(tops) > 0:
        top_ret = pred_data.loc[tops, 'PCT_RET_t1'].mean()
    else:
        top_ret = 0
        
    if len(bots) > 0:
        bot_ret = pred_data.loc[bots, 'PCT_RET_t1'].mean()
    else:
        bot_ret = 0
        
    ret = top_ret - bot_ret
    
    if stop_threshold is not None:
        
#        daily_ret_df = pd.concat([daily_xs.loc[:, tops], -daily_xs.loc[:, bots]], axis=1)
#        daily_cumret_df = ((daily_ret_df / 100 + 1).cumprod() - 1) * 100
        
        
        top_daily_ret = daily_xs.loc[:, tops]
        bot_daily_ret = daily_xs.loc[:, bots]
        
        top_daily_cumret = ((top_daily_ret / 100 + 1).cumprod() - 1) * 100
        bot_daily_cumret = ((bot_daily_ret / 100 + 1).cumprod() - 1) * 100
        
        daily_cumret_df = pd.concat([top_daily_cumret, -bot_daily_cumret], axis=1)
        
        
#        a = daily_cumret_df.copy() #%
        
        
#        top_daily_df = daily_xs.loc[:, daily_xs.columns.get_level_values(1).isin(tops)]
#        bot_daily_df = daily_xs.loc[:, daily_xs.columns.get_level_values(1).isin(bots)]
#        
#        top_ret_df = top_daily_df.loc[:, top_daily_df.columns[top_daily_df.columns.get_level_values(0) == 'PCT_RET']]
#        bot_ret_df = -bot_daily_df.loc[:, bot_daily_df.columns[bot_daily_df.columns.get_level_values(0) == 'PCT_RET']]
#        
#        topbot_daily_df = pd.concat([top_daily_df, bot_daily_df], axis=1)
#        daily_ret_df = pd.concat([top_ret_df, bot_ret_df], axis=1)
#        daily_absret_df = pd.concat([top_ret_df, -bot_ret_df], axis=1)
#        
#        
#        daily_cumret_df = ((daily_ret_df / 100 + 1).cumprod() - 1) * 100
#        daily_vol_df = topbot_daily_df.loc[:, topbot_daily_df.columns[topbot_daily_df.columns.get_level_values(0) == 'VOLUME']]

#        if stop_method == 'profolio':
#            daily_cumret_mask = (daily_cumret_df.mean(axis=1) <= -stop_threshold).cumsum() > 0
#            if daily_cumret_mask.sum() > 0:
#                stop_date = daily_cumret_mask.idxmax()
#                stopprof = daily_absret_df.loc[stop_date, :]
#                
#                
#                stock_below_limit = stopprof.index.droplevel(0).to_series()[(stopprof < -10).values].values
#                next_day_vol = daily_vol_df.shift(-1).loc[stop_date, :].droplevel(0)
#                for s in stock_below_limit:
#                    if next_day_vol[s] is np.nan:
#                        raise Exception

        if stop_method == 'individual':
            daily_cumret_mask = (daily_cumret_df <= -stop_threshold).cumsum() > 0
            for i, col in enumerate(daily_cumret_df.columns):
#                if daily_vol_df.ix[daily_cumret_mask.shift(1).fillna(False).ix[:, i].idxmax(), i] is np.nan:
#                    raise Exception
                daily_cumret_df.ix[daily_cumret_mask.ix[:, i].astype(bool).values, i] = daily_cumret_df.ix[daily_cumret_mask.ix[:, i].idxmax(), i]
                
                
            #%
#            b = daily_cumret_df.copy()
#            
#            count = daily_cumret_df.columns.size
#            not_stopped = (~daily_cumret_mask.any()).sum()
#            
#            not_stopped_return = a[a.columns[~daily_cumret_mask.any()]].iloc[-1, :].mean()
#            
#            a = a[a.columns[daily_cumret_mask.any()]]
#            b = b[b.columns[daily_cumret_mask.any()]]
#            
#            a = a.iloc[-1, :]
#            b = b.iloc[-1, :]
#            
#            recovered = (a >= b).sum()
#            not_recovered = (a < b).sum()
#            
#            
#            loss_from_recovered_return = a[a >= b].mean()
#            gain_from_not_recovered_return = a[a < b].mean()
#            
#            realized_recovered_return = b[a >= b].mean()
#            realized_not_recovered_return = b[a < b].mean()
            
            

            #%
        
        if strategy in ['L', 'S']:
            multiplier = 1
        else:
            multiplier = 2
            
        daily_prof_ret_df = daily_cumret_df.mean(axis=1) * multiplier
        
        if stop_method == 'individual':
            ret = daily_prof_ret_df[-1]
        if stop_method == 'profolio':
            daily_prof_cumret_mask = daily_prof_ret_df <= -stop_threshold
            if daily_prof_cumret_mask.any():
                ret = daily_prof_ret_df[daily_prof_cumret_mask.idxmax()]

    return ret #, count, not_stopped, recovered, not_recovered, not_stopped_return, loss_from_recovered_return, gain_from_not_recovered_return, realized_recovered_return, realized_not_recovered_return  #, profolio #%
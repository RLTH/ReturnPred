import numpy as np
import pandas as pd


old_col = ['PX_CLOSE_1D', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME', 'CHG_NET_1D', 'CHG_PCT_1D', 'TURNOVER', 'CUR_MKT_CAP', 'EQY_SH_OUT', 'EV_TO_T12M_EBITDA', 'EV_TO_T12M_SALES', 'PX_TO_BOOK_RATIO', 'PE_RATIO', 'PX_TO_SALES_RATIO', 'CAPEX_TO_DEPR_EXPN_RATIO', 'CUR_RATIO', 'LT_DEBT_TO_COM_EQY', 'GROSS_MARGIN', 'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'ASSET_TURNOVER', 'BEST_DIV_YLD', 'EQY_DVD_YLD_12M', 'EBIT_YIELD', 'CASH_FLOW_YIELD', 'FREE_CASH_FLOW_YIELD', 'EPS_GROWTH', 'RETURN_ON_INV_CAPITAL', 'ALTMAN_Z_SCORE', 'DVD_PAYOUT_RATIO', 'TOTAL_DEBT_1_YEAR_GROWTH', 'SHORT_INT_RATIO', 'VOLATILITY_20D', 'EQY_DVD_YLD_IND', 'HAS_CONVERTIBLES', 'BETA_ADJ_OVERRIDABLE', 'GICS_INDUSTRY_NAME', 'TOT_DEBT_TO_COM_EQY', 'BEST_BPS', 'CAPITAL_EXPEND', 'BEST_TARGET_PRICE', 'IS_COGS_TO_FE_AND_PP_AND_G', 'TOT_COMMON_EQY', 'BS_CUR_ASSET_REPORT', 'BS_CUR_LIAB', 'CF_DEPR_AMORT', 'EQY_DPS', 'BEST_DPS', 'EBITDA', 'BEST_EBITDA', 'IS_EPS', 'BEST_EPS', 'ENTERPRISE_VALUE', 'CF_FREE_CASH_FLOW', 'BEST_ESTIMATE_FCF', 'IS_INC_BEF_XO_ITEM', 'IS_INT_EXPENSE', 'BS_INVENTORIES', 'TOTAL_INVESTED_CAPITAL', 'CF_ISSUE_COM_STOCK', 'BS_LT_BORROW', 'NET_INCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NET_OPERATING_ASSETS', 'CF_CASH_FROM_OPER', 'IS_OPERATING_EXPN', 'IS_OPER_INC', 'BEST_PX_BPS_RATIO', 'PX_TO_CASH_FLOW', 'BEST_PX_CPS_RATIO', 'BEST_PE_RATIO', 'BEST_PEG_RATIO', 'BS_GROSS_FIX_ASSET', 'BEST_ANALYST_RATING', 'IS_RD_EXPEND', 'BEST_ROA', 'BEST_ROE', 'SALES_REV_TURN', 'BEST_SALES', 'BS_TOT_ASSET', 'SHORT_AND_LONG_TERM_DEBT', 'BS_TOT_LIAB2', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CURR_ENTP_VAL', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP', 'MOV_AVG_30D', 'MOV_AVG_60D', 'MOV_AVG_200D', 'MOV_AVG_5D', 'MOV_AVG_10D', 'MOV_AVG_20D', 'MOV_AVG_40D', 'MOV_AVG_50D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'WACC', 'APPLIED_BETA', 'LT_DEBT_TO_TOT_EQY', 'BEST_NAV', 'BEST_NET_DEBT', 'BEST_CPS', 'BEST_CAPEX', 'BEST_CURRENT_PROFIT', 'BEST_DEPRECIATION', 'BEST_EBIT', 'BEST_NET_INCOME', 'BEST_OPP', 'BEST_PTP']
new_col = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'FY1_YLD', 'TRL_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_DEBT', 'SHORT_COV', 'RETVOL', 'DY', 'CONVIND', 'BETA', 'IND_CODE', 'LEV', 'BPS_FY1', 'CAPEX', 'CLOSE_FY1', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY1', 'EBITDA', 'EBITDA_FY1', 'EPS', 'EPS_FY1', 'EV', 'FCF', 'FCF_FY1', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY1', 'PCF', 'PCPS_FY1', 'PE_FY1', 'PEG_FY1', 'PPEGT', 'RATING_FY1', 'RD_EXP', 'ROA_FY1', 'ROE_FY1', 'SALES', 'SALES_FY1', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CURR_ENTP_VAL', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP', 'MOV_AVG_30D', 'MOV_AVG_60D', 'MOV_AVG_200D', 'MOV_AVG_5D', 'MOV_AVG_10D', 'MOV_AVG_20D', 'MOV_AVG_40D', 'MOV_AVG_50D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'WACC', 'APPLIED_BETA', 'LT_DEBT_TO_TOT_EQY', 'NAV_FY1', 'DEBT_FY1', 'CPS_FY1', 'CAPEX_FY1', 'PROFIT_FY1', 'DEPRECIATION_FY1', 'EBIT_FY1', 'INCOME_FY1', 'OPP_FY1', 'PTP_FY1']
new_col_index = ['INDEX_' + col for col in new_col]
tofill_col = ['EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'IND_CODE', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CURR_ENTP_VAL', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP', 'MOV_AVG_30D', 'MOV_AVG_60D', 'MOV_AVG_200D', 'MOV_AVG_5D', 'MOV_AVG_10D', 'MOV_AVG_20D', 'MOV_AVG_40D', 'MOV_AVG_50D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'WACC', 'APPLIED_BETA', 'LT_DEBT_TO_TOT_EQY', 'NAV_FY1', 'DEBT_FY1', 'CPS_FY1', 'CAPEX_FY1', 'PROFIT_FY1', 'DEPRECIATION_FY1', 'EBIT_FY1', 'INCOME_FY1', 'OPP_FY1', 'PTP_FY1', 'NAV_FY0', 'DEBT_FY0', 'CPS_FY0', 'CAPEX_FY0', 'PROFIT_FY0', 'DEPRECIATION_FY0', 'EBIT_FY0', 'INCOME_FY0', 'OPP_FY0', 'PTP_FY0', 'NAV_FY2', 'DEBT_FY2', 'CPS_FY2', 'CAPEX_FY2', 'PROFIT_FY2', 'DEPRECIATION_FY2', 'EBIT_FY2', 'INCOME_FY2', 'OPP_FY2', 'PTP_FY2', 'GICS_SECTOR_NAME', 'GICS_INDUSTRY_GROUP_NAME', 'GICS_SUB_INDUSTRY_NAME']

col_dict = dict(zip(old_col, new_col))

final_col = ['HSI', 'HSCI', 'HSCEI', 'PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'IND_CODE', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CURR_ENTP_VAL', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP', 'MOV_AVG_30D', 'MOV_AVG_60D', 'MOV_AVG_200D', 'MOV_AVG_5D', 'MOV_AVG_10D', 'MOV_AVG_20D', 'MOV_AVG_40D', 'MOV_AVG_50D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'WACC', 'APPLIED_BETA', 'LT_DEBT_TO_TOT_EQY', 'NAV_FY1', 'DEBT_FY1', 'CPS_FY1', 'CAPEX_FY1', 'PROFIT_FY1', 'DEPRECIATION_FY1', 'EBIT_FY1', 'INCOME_FY1', 'OPP_FY1', 'PTP_FY1', 'NAV_FY0', 'DEBT_FY0', 'CPS_FY0', 'CAPEX_FY0', 'PROFIT_FY0', 'DEPRECIATION_FY0', 'EBIT_FY0', 'INCOME_FY0', 'OPP_FY0', 'PTP_FY0', 'NAV_FY2', 'DEBT_FY2', 'CPS_FY2', 'CAPEX_FY2', 'PROFIT_FY2', 'DEPRECIATION_FY2', 'EBIT_FY2', 'INCOME_FY2', 'OPP_FY2', 'PTP_FY2']


def data_formatting(df):
    
    df = df.drop_duplicates(['date', 'ticker', 'field'])
    
    df = df.set_index(['date', 'ticker', 'field']).unstack(level='field')
    df.columns = df.columns.droplevel(0)
    df.index.names = ['DATE', 'SYMBOL']
    df.columns.names = ['FIELD']
    
    # col rename
    df = df.rename(columns=col_dict)
    
    # fillna
    tofill_col_intersection = df.columns[df.columns.isin(tofill_col)]
    df[tofill_col_intersection] = df[tofill_col_intersection].groupby(level='SYMBOL').fillna(method='ffill')
    
    # index DATE level to datetime
    df.index = df.index.set_levels([
            pd.to_datetime(df.index.levels[0]),
            df.index.levels[1],
            ])

    return df



mth_symbol = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(level='SYMBOL')]
mth_ind = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(key='IND_CODE')]
mth = [pd.Grouper(freq='M', level='DATE'), ]
symbol = [pd.Grouper(level='SYMBOL'), ]


################################################################################
print('reading daily...')


daily = pd.read_csv('../_data/_daily/bdh.csv')

daily = daily.drop_duplicates(['date', 'ticker', 'field'])

daily = data_formatting(daily)

daily['1base_pct_ret'] = daily['PCT_RET'] / 100 + 1
mth_return_from_daily = (daily['1base_pct_ret'].groupby(mth_symbol).prod() - 1) * 100


################################################################################
print('reading mkt...')

indexes = ['HSI', 'HSCI', 'HSCEI']

mkt_data = []

for index in indexes:
    df = pd.read_csv('../_data/_index/{}.csv'.format(index), header=[0, 1, 2])
    df.columns = df.columns.droplevel([0, 2])[1: ].insert(0, 'DATE')
    df = df.set_index('DATE')
    df.index = pd.to_datetime(df.index)

    prefix = '{}_'.format(index)
    new_col_index = [prefix + col for col in new_col]
    col_dict_index = dict(zip(old_col, new_col_index))
    df = df.rename(columns=col_dict_index)
    
    mkt_data.append(df)
    
mkt_df = pd.concat(mkt_data, axis=1)

rf_df = pd.read_csv('../_data/_rf/bdh.csv')
rf_df = data_formatting(rf_df)
rf_df = rf_df.reset_index('SYMBOL', drop=True)
rf_df = rf_df.rename(columns={'CLOSE': 'RF_pa'})

mkt_df['RF'] = ((rf_df['RF_pa'] / 100 + 1) ** (1/12) - 1) * 100


################################################################################
print('reading est...')


est_datas = []
for part in [1, 2]:
    for yr in [0, 1, 2]:
        path = '../_data/_est/fy{}_{}.csv'.format(yr, part)
        data = pd.read_csv(path)
        data = data_formatting(data)
        data.columns = data.columns.str.replace('FY1', 'FY{}'.format(yr))
        est_datas.append(data)

est_df = pd.concat(est_datas, axis=1)


################################################################################
print('reading bdh...')


paths = ['../_data/1/bdh.csv',
         '../_data/2/bdh.csv',
         '../_data/3/bdh.csv',
         '../_data/4/bdh.csv',
         '../_data/5/bdh.csv',
         '../_data/6/bdh.csv',
         '../_data/7/bdh.csv',
         '../_data/8/bdh.csv',
         ]


datas = []
for path in paths:
    data = pd.read_csv(path)
    data = data_formatting(data)
    datas.append(data)

paths = ['../_data/_ref/ref0.csv',
         '../_data/_ref/ref1.csv',
         '../_data/_ref/ref2.csv',
         '../_data/_ref/ref3.csv',
         ]

refs = []
for path in paths:
    data = pd.read_csv(path)
    data = data_formatting(data)
    refs.append(data)
ref = pd.concat(refs, axis=0)
datas.append(ref)


df = pd.concat(datas, axis=1)
df = df.join(est_df)

df = df.loc[:,~df.columns.duplicated()]

################################################################################
print('reading index historical components...')

def read_index_histcom(path):
    df = pd.read_csv(path)
    df = df[['date', 'name', 'value', 'position']]
    df = df.groupby('date').apply(lambda x: x.pivot(index='position', columns='name', values='value'))
    df = df.reset_index(drop=False)
    df.columns = ['DATE', 'POSITION', 'SYMBOL', 'WEIGHT']
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['SYMBOL'] = df['SYMBOL'] + ' Equity'
    df = df.set_index(['DATE', 'SYMBOL'])
    return df

hsi_df = read_index_histcom('../_data/_index/hsi_histcom.csv')
hsci_df = read_index_histcom('../_data/_index/hsci_histcom.csv')
hscei_df = read_index_histcom('../_data/_index/hscei_histcom.csv')



################################################################################
print('doing historical components...')

df['HSI'] = df.index.isin(hsi_df.index)
df['HSCI'] = df.index.isin(hsci_df.index)
df['HSCEI'] = df.index.isin(hscei_df.index)


########## DB paper variables  ############################################
print('doing DB...')

df['BARRY_R'] = df['GROSS_MARGIN'] / df['OPEXP']
df['CFY_FY0'] = 1 / df['PCPS_FY0']
df['EBP_MDN'] = 1 / df['PB_FY0']
df['EPSY_FY0'] = 1 / df['PE_FY0']
df['EPSY_FY1_AVG'] = 1 / df['PE_FY1']
df['EPSY_FY2_AVG'] = 1 / df['PE_FY2']
df['RTN1260D'] = df['CLOSE'] / df['CLOSE'].groupby(symbol).shift(60) - 1 
df['RTN21D'] = df['CLOSE'] / df['CLOSE'].groupby(symbol).shift(1) - 1 
df['RTN252D'] = df['CLOSE'] / df['CLOSE'].groupby(symbol).shift(12) - 1 
df['RTN12_1M'] = df['RTN252D'] - df['RTN21D']

#MA_CO_15_36W
MA15 = daily['CLOSE'].groupby(symbol, group_keys=False).rolling(window=15*5).mean()
MA36 = daily['CLOSE'].groupby(symbol, group_keys=False).rolling(window=36*5).mean()
daily['MA_CO_15_36W'] = MA15 - MA36
df['MA_CO_15_36W'] = daily['MA_CO_15_36W'].groupby(mth_symbol).tail(1)

#REAL_VOL_1YD
daily['LOGRET'] = np.log( daily['PCT_RET'] / 100 + 1 )
daily['REAL_VOL_1YD'] = 100 * np.sqrt( 252 / 252 * (daily['LOGRET'] ** 2).groupby(symbol, group_keys=False).rolling(window=252).sum() )
df['REAL_VOL_1YD'] = daily['REAL_VOL_1YD'].groupby(mth_symbol).tail(1)

#SKEW_1YD
daily['SKEW_1YD'] = daily['LOGRET'].groupby(symbol, group_keys=False).rolling(252).skew()
df['SKEW_1YD'] = daily['SKEW_1YD'].groupby(mth_symbol).tail(1)

df['EXP_YLD'] = df['DPS_FY1'] / df['CLOSE']
df['P_52WHI'] = df['HIGH'].groupby(symbol, group_keys=False).rolling(window=12).max()
df['P_52WLO'] = df['LOW'].groupby(symbol, group_keys=False).rolling(window=12).min()
df['CHG_SHARES'] = ( df['ISSUED_SHARE'] / df['ISSUED_SHARE'].groupby(symbol, group_keys=False).shift(12) ) - 1
df['RNOA'] = df['OPINC'] / df['NOA']
df['CFROE_CF'] = df['OPCF'] / df['COMEQ']
df['CFROC_CF'] = df['FCF']/df['INV_CAP'] 


########## ML paper variables  ############################################
print('doing ML...')

df['RET'] = df['PCT_RET'] / 100 + 1
daily['RET'] = daily['PCT_RET'] / 100 + 1

df['MOM1M'] = df['RET'].groupby(symbol).shift(1)
df['MOM12M'] = df['RET'].groupby(symbol, group_keys=False).rolling(12 - 1).apply(lambda x: x.prod(), raw=True).shift(2)

#INDMOM
INDMOM = df.groupby(mth_ind)['MOM12M'].mean().rename('INDMOM')
df['INDMOM'] = df.join(INDMOM, on=['DATE', 'IND_CODE'])['INDMOM']

df['MVEL1'] = np.log(df['MKT_CAP'])
df['MAXRET'] = daily['PCT_RET'].groupby(mth_symbol).max()
df['MOM6M'] = df['RET'].groupby(symbol, group_keys=False).rolling(6 - 1).apply(lambda x: x.prod(), raw=True).shift(2)
df['CHMOM'] = df['MOM6M'] - df['MOM6M'].groupby(symbol).shift(6)
df['DOLVOL'] = np.log( df['CLOSE'].groupby(symbol).shift(2) * df['VOLUME'].groupby(symbol).shift(2) )
df['SP'] = 1 / df['PSALES']
df['TURN'] = df['VOLUME'].groupby(symbol, group_keys=False).rolling(3).mean().shift(1) / df['ISSUED_SHARE']

#NINCR
INCR = ( df['INC_BXO'] - df['INC_BXO'].groupby(symbol).shift(1) ) >= 0
NINCR = INCR.groupby(symbol, group_keys=False).apply(lambda y: y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1))
#NINCR[NINCR > 8] = 8
df['NINCR'] = NINCR

df['RD_MVE'] = df['RD_EXP'] / df['MKT_CAP']
df['AGR'] = df['TOT_ASSET'] / df['TOT_ASSET'].groupby(symbol).shift(3) - 1 #lag 1 but quarterly data

#STD_TURN
daily['VOL_SHROUT'] = daily['VOLUME'] / daily['ISSUED_SHARE']
df['STD_TURN'] = daily['VOL_SHROUT'].groupby(mth_symbol).std()

df['INVEST'] = ( (df['PPEGT']-df['PPEGT'].groupby(symbol).shift(3)) + (df['INV']-df['INV'].groupby(symbol).shift(3)) ) / df['TOT_ASSET'].groupby(symbol).shift(3)

#ILL
daily['ILL'] = np.abs(daily['RET']) / ( np.abs(daily['CLOSE']) * daily['VOLUME'] )
df['ILL'] = daily['ILL'].groupby(mth_symbol).std()

df['EP'] = 1 / df['PE_T12MOB']
df['CONVIND'] = (df['CONVIND'] == 'Y').astype(int)
df['CHINV'] = ( df['INV'] - df['INV'].groupby(symbol).shift(3) ) / (( df['TOT_ASSET'] + df['TOT_ASSET'].groupby(symbol).shift(3) ) / 2) #lag 1 but quarterly data
df['MOM36M'] = df['RET'].groupby(symbol, group_keys=False).rolling(24).apply(lambda x: x.prod(), raw=True).shift(13) 


#PS
df['PS'] = (df['NETINCOME'] > 0).astype(int) +\
 (df['OPCF'] > 0).astype(int) +\
 ((df['NETINCOME']/df['TOT_ASSET']) > ((df['NETINCOME']/df['TOT_ASSET']).groupby(symbol).shift(1))).astype(int) +\
 (df['OPCF'] > df['NETINCOME']).astype(int) +\
 ((df['LONGDEBT']/df['TOT_ASSET']) < ((df['LONGDEBT']/df['TOT_ASSET']).groupby(symbol).shift(1))).astype(int) +\
 ((df['CUR_ASSET']/df['CUR_LIAB']) > ((df['CUR_ASSET']/df['CUR_LIAB']).groupby(symbol).shift(1))).astype(int) +\
 ((1 - df['COGS']/df['SALES']) > ((1 - df['COGS']/df['SALES']).groupby(symbol).shift(1))).astype(int) +\
 ((df['SALES']/df['TOT_ASSET']) > ((df['SALES']/df['TOT_ASSET']).groupby(symbol).shift(1))).astype(int) +\
 (df['ISS_COMSTOCK'] == 0).astype(int)   

#ZEROTRADE
daily['VOLZERO'] = (daily['VOLUME'] == 0).astype(int)
df['COUNTZERO'] = daily['VOLZERO'].groupby(mth_symbol).sum()
df['NDAYS'] = daily['PCT_RET'].groupby(mth_symbol).count()
df['ZEROTRADE'] = (df['COUNTZERO'] + ((1/df['TURNOVER'])/480000))*21/df['NDAYS']

df['LGR'] = df['LONGDEBT'] / df['LONGDEBT'].groupby(symbol).shift(3) - 1 #lag 1 but quarterly data
df['DEPR'] = df['DEP'] / df['PPEGT']
df['BM'] = 1 / df['PB']
df['BETASQ'] = df['BETA'] ** 2
df['OPERPROF'] = ( df['OPINC'] - df['INTEXP'] ) / df['COMEQ']

#BM_IA
BM_IA = df.groupby(mth_ind)['BM'].mean().rename('BM_IA')
df['BM_IA'] = df['BM'] - df.join(BM_IA, on=['DATE', 'IND_CODE'])['BM_IA']



################################################################################
print('checking and cleaning...')

mth_return = df['PCT_RET']
mistake_mask = (abs(mth_return_from_daily - mth_return) > 0.1)
if mistake_mask.sum() > 0:
    print(mistake_mask[mistake_mask])
    mistake_idx = mistake_mask[mistake_mask].index

df.loc[mistake_idx, 'PCT_RET'] = mth_return_from_daily[mistake_idx]

df = df.loc[:, df.columns.isin(final_col)]
daily_prices = daily[['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'PCT_RET', 'VOLUME']]


################################################################################
print('saving...')

daily_prices.to_csv('../_data/daily.csv')
df.to_csv('../_data/data.csv')
mkt_df.to_csv('../_data/mkt.csv')


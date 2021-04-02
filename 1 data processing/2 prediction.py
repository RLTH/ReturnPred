
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
#import seaborn as sns

#from sklearn.preprocessing import scale

#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


#import sklearn.linear_model as skl_lm
#from sklearn.metrics import mean_squared_error, r2_score
#import statsmodels.api as sm
#import statsmodels.formula.api as smf


#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.externals.six import StringIO
#from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
#from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
#from sklearn.ensemble import AdaBoostRegressor

#from statistics import mean
from sklearn.neural_network import MLPRegressor

#%matplotlib inline
plt.style.use('seaborn-white')

import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore')

################################################################################

no = 94

lag_period = 12
rank_x = True #%
rank_y = True
binary_y_return_sign = False
binary_y_topbot = False
topbotcount = False

warmstart = False #%
#np.random.seed(seed=0)

#methods = ['LinReg', 'GradBoost', 'RandForest', 'NerualNet', 'XGBLin', 'XGBLog',\
#'postLasso', 'XGBLog_sel', 'ElasticNet', 'XGBLin_sel']

method = 'ElasticNet' #%

ADV_filter = True

generate_corr = False
generate_modelinfo = True



if not warmstart:

    ######################################## top factors from single factor ranking on all stocks
#    x_var_selection = \
#    ['EPSY_T12MOB', 'PE_T12MOB', 'EP', 'BM', 'RTN252D', 'NEWS_POS_SENTIMENT_COUNT', 'PB', 'NINCR', 'CFY_IS', 'RTN12_1M', 'DY', 'YOY_EPS_G', 'PEG_FY0', 'FCF_YLD', 'CF_DISP_FIX_ASSET', 'BM_IA', 'TRL_YLD', 'CF_CASH_FROM_INV_ACT', 'CHG_SHARES', 'INV', 'OPERPROF', 'DEPR', 'CF_INCR_LT_BORROW', 'RNOA', 'PEG_FY1', 'COGS', 'STD_TURN', 'MOV_AVG_120D', 'CAPEX_FY2', 'IS_INC_TAX_EXP', 'BS_ACCT_NOTE_RCV', 'SP', 'RD_EXP', 'MOV_AVG_100D', 'MOM12M', 'CF_CASH_FROM_FNC_ACT', 'CUR_ASSET', 'RD_MVE', 'TOT_ASSET', 'CF_DVD_PAID', 'CFROC_CF', 'PCPS_FY0', 'RSI_9D', 'CFROE_CF', 'PAYOUT_OEPS', 'COMEQ', 'PCF', 'BS_CASH_NEAR_CASH_ITEM', 'CAPEX', 'CFY_FY0', 'MKT_CAP', 'MVEL1', 'ROE', 'PE_FY1', 'DOLVOL', 'CURR_ENTP_VAL', 'ISSUED_SHARE', 'P_52WLO', 'PB_FY0', 'BS_LT_INVEST', 'CF_DECR_INVEST', 'INV_CAP', 'CF_DECR_CAP_STOCK', 'EBITDA_EV', 'WACC', 'OPINC', 'CF_INCR_INVEST', 'TURNOVER', 'CONVIND', 'MOV_AVG_5D', 'PCPS_FY1', 'BS_ACCT_PAYABLE', 'OPCF', 'MOV_AVG_10D', 'CF_CAP_EXPEND_PRPTY_ADD', 'MOV_AVG_60D', 'IS_NET_NON_OPER_LOSS', 'PS', 'BS_SH_OUT', 'ROIC', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'LOW', 'PRETAX_INC', 'SAL_EV', 'INCOME_FY0', 'EXP_YLD', 'EV', 'TOT_LIAB', 'CF_NET_INC', 'ROA_FY2', 'AGR', 'ROE_FY0', 'NETINCOME', 'MOV_AVG_40D', 'EBP_MDN', 'CLOSE', 'INTEXP', 'GROSS_PROFIT', 'MOV_AVG_20D', 'PE_FY0', 'PSALES', 'LONGDEBT', 'CHINV', 'PEG_FY2', 'EPS_FY2', 'ZEROTRADE', 'P_52WHI', 'ILL', 'DEPRECIATION_FY2', 'CF_REIMB_LT_BORROW', 'SALES', 'MOV_AVG_30D', 'INDMOM', 'MA_CO_15_36W', 'BS_TOT_NON_CUR_ASSET', 'TOT_DEBT', 'ROA_FY0', 'PB_FY2', 'DPS_FY2', 'PE_FY2', 'OPEN', 'INC_BXO', 'FY2_YLD', 'RATING_FY0', 'OPP_FY2', 'RATING_FY1', 'CUR_LIAB', 'FCF_FY1', 'FY1_YLD', 'DPS', 'PB_FY1', 'FCF', 'RSI_14D', 'MOV_AVG_200D', 'PRE', 'TWITTER_POS_SENTIMENT_COUNT', 'ROE_FY2', 'EBITDA', 'CHMOM', 'OPP_FY0', 'HIGH', 'EBIT_FY2', 'NAV_FY1', 'FY0_YLD', 'PCT_RET', 'INCOME_FY1', 'MOV_AVG_50D', 'BPS_FY0', 'EPS_FY0', 'INCOME_FY2', 'SALES_FY0', 'EPS_FY1', 'NAV_FY0', 'EPSY_FY0', 'EPSY_FY1_AVG', 'RTN21D', 'RATING_FY2', 'CLOSE_FY2', 'MOM1M', 'SALES_FY1', 'CURRENT_R', 'FCF_FY0', 'DEBT_EQUITY', 'ROE_FY1', 'BS_ST_BORROW', 'BS_NET_FIX_ASSET', 'PTP_FY2', 'CPS_FY2', 'SALES_FY2', 'NEWS_SENTIMENT_DAILY_AVG', 'RETVOL', 'CF_INCR_CAP_STOCK', 'VOLUME', 'NAV_FY2', 'BPS_FY2', 'ROA', 'LT_DEBT_TO_TOT_EQY', 'PTP_FY0', 'TURN', 'BS_TOT_EQY', 'EBITDA_FY2', 'RSI_30D', 'NEWS_SENTIMENT_DAILY_MAX', 'OPP_FY1', 'CLOSE_FY0', 'BPS_FY1', 'DEP', 'INVEST', 'OPEXP', 'EPS', 'GROSS_MARGIN', 'EBITDA_FY0', 'MOM6M', 'DPS_FY1', 'EBIT_FY0', 'LEV', 'ROA_FY1', 'PPEGT', 'PTP_FY1', 'CLOSE_FY1', 'BETASQ', 'DEBT_FY0', 'CAPEX_DEP_12M', 'SAL_TA', 'CPS_FY0', 'BARRY_R', 'DEBT_FY2', 'CPS_FY1', 'DEPRECIATION_FY0', 'DEPRECIATION_FY1', 'DPS_FY0', 'MOM36M', 'APPLIED_BETA', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'EPSY_FY2_AVG', 'MAXRET', 'PRICE_CHG', 'CHG_DEBT', 'PROFIT_FY0', 'DEBT_FY1', 'TWITTER_SENTIMENT_DAILY_MIN', 'ISS_COMSTOCK', 'RTN1260D', 'CAPEX_FY0', 'SKEW_1YD', 'CAPEX_FY1', 'TWITTER_NEG_SENTIMENT_COUNT', 'PCPS_FY2', 'PROFIT_FY2', 'EBIT_FY1', 'BETA', 'FCF_FY2', 'LGR', 'EBITDA_FY1', 'REAL_VOL_1YD', 'ALTMAN', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_AVG', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'PROFIT_FY1', 'SHORT_COV', 'NEWS_NEG_SENTIMENT_COUNT']
#    
#    x_var_selection = x_var_selection[:20] #%
    
    

    ######################################## All variables including dummies excluding accounting
    x_var_selection = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'IND_CODE', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN']
    
    ######################################## All variables including dummies
#    x_var_selection = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'IND_CODE', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CURR_ENTP_VAL', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP', 'MOV_AVG_30D', 'MOV_AVG_60D', 'MOV_AVG_200D', 'MOV_AVG_5D', 'MOV_AVG_10D', 'MOV_AVG_20D', 'MOV_AVG_40D', 'MOV_AVG_50D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'WACC', 'APPLIED_BETA', 'LT_DEBT_TO_TOT_EQY', 'NAV_FY1', 'DEBT_FY1', 'CPS_FY1', 'CAPEX_FY1', 'PROFIT_FY1', 'DEPRECIATION_FY1', 'EBIT_FY1', 'INCOME_FY1', 'OPP_FY1', 'PTP_FY1', 'NAV_FY0', 'DEBT_FY0', 'CPS_FY0', 'CAPEX_FY0', 'PROFIT_FY0', 'DEPRECIATION_FY0', 'EBIT_FY0', 'INCOME_FY0', 'OPP_FY0', 'PTP_FY0', 'NAV_FY2', 'DEBT_FY2', 'CPS_FY2', 'CAPEX_FY2', 'PROFIT_FY2', 'DEPRECIATION_FY2', 'EBIT_FY2', 'INCOME_FY2', 'OPP_FY2', 'PTP_FY2']
 
    ######################################## accounting variables
#    x_var_selection = ['COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'FCF', 'INTEXP', 'INV', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'OPCF', 'OPEXP', 'OPINC', 'PPEGT', 'RD_EXP', 'SALES', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP']
    
    

    ####################################### GP
#    x_var_selection = ['BM', 'OPERPROF'] # BM*OPERPROF
#    x_var_selection = ['ILL', 'MKT_CAP'] # ILL/MKT_CAP
#    x_var_selection = ['OPERPROF', 'PB'] # OPERPROF/PB
#    x_var_selection = ['DOLVOL', 'EPSY_FY0'] # -DOLVOL + EPSY_FY0

    
    
    
    
    ################################################################################
    
    def prediction(train_x, train_y, test_x, method):
        model_info = None
    
        if method == 'LinReg':
            model = LinearRegression()
        elif method == 'GradBoost':
            model = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.001, random_state=1)
        elif method == 'RandForest':
            model = RandomForestRegressor(max_leaf_nodes=48, max_features='sqrt', n_estimators=100, random_state=1)
        elif method == 'NerualNet':
            model = MLPRegressor(hidden_layer_sizes=(30, 15),random_state=0, learning_rate_init = 0.01)
            train_y = train_y.flatten()
        elif method == 'XGBLin': #%
    #        model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.25, learning_rate = 0.002, max_depth = 5, alpha=50, n_estimators = 1000,random_state=0)
            model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.25, learning_rate = 0.001, max_depth = 2, alpha=0, n_estimators = 1000,random_state=0)
        elif method == 'XGBLog': 
            model = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.25, learning_rate = 0.002, max_depth = 5, alpha=50, n_estimators = 1000,random_state=0)
        elif method == 'postLasso':
            alphas = np.arange(0, 2.01, 0.02)
            cv = LassoCV(alphas=alphas,cv=50,random_state=0)
            cv.fit(train_x, train_y)
            model = Lasso(alpha=cv.alpha_,random_state=0)
            model.fit(train_x,train_y)
            print(cv.alpha_)
#            print(cv.mse_path_)
            model_info = {'alpha': cv.alpha_, 'coef': model.coef_}
    
            lasso_position=np.where(abs(model.coef_)>(10**(-5)))
            if len(lasso_position[0])==0:
                print('No selection')
                lasso_position=[0]
            train_x=train_x[:,lasso_position[0]]
            test_x=test_x[:,lasso_position[0]]
            if len(test_x.shape)==1:
                train_x=train_x.reshape(-1,1)
                test_x=test_x.reshape(-1,1)
            model = LinearRegression()
        elif method == 'XGBLog_sel':
            model = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.25, learning_rate = 0.002, max_depth = 5, alpha=50, n_estimators = 1000,random_state=0)
            model.fit(train_x,train_y)
            selected_var_pos = np.argsort(model.feature_importances_)[-10:]
            train_x=train_x[:,selected_var_pos]
            test_x=test_x[:,selected_var_pos]
            model = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.25, learning_rate = 0.002, max_depth = 5, alpha=50, n_estimators = 1000,random_state=0)
        elif method == 'XGBLin_sel':
            model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.25, learning_rate = 0.001, max_depth = 2, alpha=0, n_estimators = 1000,random_state=0)
            model.fit(train_x,train_y)
            selected_var_pos = np.argsort(model.feature_importances_)[-10:] #%
            train_x=train_x[:,selected_var_pos]
            test_x=test_x[:,selected_var_pos]
            model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.25, learning_rate = 0.001, max_depth = 2, alpha=0, n_estimators = 1000,random_state=0)
        elif method == 'ElasticNet':
#            alphas=np.array(range(500))*0.01
            alphas = np.arange(0, 2.01, 0.02)
            cv = ElasticNetCV(alphas=alphas,l1_ratio=[.1, .5, .7, .9, .95, .99, 1],cv=50,random_state=0)
            cv.fit(train_x, train_y)
            print(cv.alpha_, cv.l1_ratio_)
            model = ElasticNet(alpha=cv.alpha_,l1_ratio=cv.l1_ratio_)
            model_info = {'alpha': cv.alpha_, 'l1_ratio': cv.l1_ratio_}
    
        model.fit(train_x,train_y)
        pred_y = model.predict(test_x)
        pred_y = pred_y.reshape(-1,1)
    
        if generate_modelinfo:
            if method == 'ElasticNet':
                model_info['coef'] = model.coef_
        else:
            model_info = None
        
        return pred_y, model_info




    #######################################################################
    #                            Data Loading                             #
    #######################################################################
    
    filename = '../_data/data.csv'
    raw_df = pd.read_csv(filename)
    
    if generate_corr or ADV_filter:
        daily_filename = '../_data/daily.csv'
        daily = pd.read_csv(daily_filename) #, usecols=['SYMBOL', 'DATE', 'PCT_RET']
        daily['DATE'] = pd.to_datetime(daily['DATE'])
        daily = daily.set_index(['DATE', 'SYMBOL'])
    
    print('data loaded')
    
    # Putting back the missing rows from (dates * stocks)
#    raw_df = raw_df.dropna(subset=['SYMBOL', 'PCT_RET', 'VOLUME'])
    raw_df = raw_df.set_index(['DATE', 'SYMBOL']).sort_index()
    stocks = raw_df.index.get_level_values('SYMBOL').drop_duplicates()
    dates = raw_df.index.get_level_values('DATE').drop_duplicates()
    raw_df = raw_df.reindex(pd.MultiIndex.from_product((dates, stocks), names=('DATE', 'SYMBOL')), fill_value=np.nan).sort_index()
    
    # One hot encoding for industry
    industry_column_name = 'IND_CODE'
    if industry_column_name in x_var_selection:
        dummies = pd.get_dummies(raw_df, columns=[industry_column_name])
        raw_df = pd.concat([dummies, raw_df['IND_CODE']], axis=1) 
        allvariables = list(raw_df.columns)
        industryvariable = [x for x in allvariables if industry_column_name in x and industry_column_name != x]
        x_var_selection = x_var_selection + industryvariable
        x_var_selection.remove(industry_column_name)
    
    #######################################################################
    #                            Data Cleaning                            #
    #######################################################################
    
    # x variables are 1 month lag
    df = raw_df.groupby(level='SYMBOL').shift(1)
    df['PCT_RET_t1'] = raw_df['PCT_RET']
    
    # suspended stocks are filtered out
    df['invalid'] = ( df['VOLUME'].isna() | df['PCT_RET'].isna() ).groupby(level='SYMBOL', group_keys=False).rolling(3).sum()
    df['invalid'] = df['invalid'].astype(bool)    
    df = df[~df['invalid']]
    
    
    # correcting survivial bias
    df = df[(df['HSI'] == True) | (df['HSCI'] == True) | (df['HSCEI'] == True)]
    
    
    # ADV and MKT_CAP filtering
    if ADV_filter:
        adv_threshold = 4e7
        mktcap_threshold = 7.78e3
        
        mth_symbol = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(level='SYMBOL')]
        
        adv = (daily['CLOSE'] * daily['VOLUME']).groupby(mth_symbol).mean()
        mktcap = df['MKT_CAP']
        adv_date_mask = adv.index.get_level_values('DATE') <= '2006-12-31'
        mktcap_date_mask = mktcap.index.get_level_values('DATE') <= '2006-12-31'
        
        # The thresholds are halved before 2007 to allow for larger sample size
        adv_mask = adv > ( adv_date_mask * (adv_threshold / 2) + (~adv_date_mask) * adv_threshold )
        mktcap_mask = mktcap > ( mktcap_date_mask * (mktcap_threshold / 2) + (~mktcap_date_mask) * mktcap_threshold )
                
        adv_mask.index = adv_mask.index.set_levels([
                adv_mask.index.levels[0].strftime('%Y-%m-%d').astype(object),
                adv_mask.index.levels[1],
                ])
        
        df = df[adv_mask & mktcap_mask]

    
    
    # filling missing value with mean from valid stocks
    df = df.groupby(level='DATE', group_keys=False).transform(lambda x: x.fillna(x.mean()))
    
    
    print('df cleaned')
    
    # ranking varibles
    df['PCT_RET_t1_rank'] = df['PCT_RET_t1'].groupby(level='DATE').rank(ascending=1)
    df['PCT_RET_t1_rank_rev'] = df['PCT_RET_t1'].groupby(level='DATE').rank(ascending=0)
    
    if rank_x:
        df[x_var_selection] = df[x_var_selection].groupby(level='DATE').rank(ascending=0)
    
    if binary_y_topbot:
        df['y'] = df['PCT_RET_t1_rank_rev'] <= topbotcount
    elif binary_y_return_sign:
        df['y'] = df['PCT_RET_t1'] >= 0
    elif rank_y:
        df['y'] = df['PCT_RET_t1_rank']
    else:
        df['y'] = df['PCT_RET_t1']
    
    valid_dates = dates[lag_period:]
    
    
    meta = pd.DataFrame(index = dates)
    meta.index = pd.to_datetime(meta.index)
    meta['n'] = df.groupby(level='DATE').apply(lambda x: x.index.size)
    meta['stock_count'] = df.groupby(level='DATE').apply(lambda x: x.index.unique().size)
    meta['valid_date'] = meta.index.isin(valid_dates)


    
    print('df done')

#######################################################################
#                               Main Loop                             #
#######################################################################
    
hist_corrs = {}
model_infos = {}

#for (date, train_data), (date, test_data) in zip(train_df.groupby(level='DATE'), test_df.groupby(level='DATE')):

for date0, date in zip(dates, valid_dates):
    train_data = df[(df.index.get_level_values('DATE') >= date0) & (df.index.get_level_values('DATE') < date)]
    test_data = df[(df.index.get_level_values('DATE') == date)]
    
#    if date == '2004-01-31':
#        import sys
#        sys.exit()
#    else:
#        pass
    
    if topbotcount > 0:
        train_data = train_data.loc[(train_data['PCT_RET_t1_rank'] <= topbotcount) | (train_data['PCT_RET_t1_rank_rev'] <= topbotcount), :]

    train_y = train_data['y'].values.reshape(-1,1)
    test_y = test_data['y'].values.reshape(-1,1)
    train_data = train_data[x_var_selection].replace([np.inf, -np.inf], np.nan)
    test_data = test_data[x_var_selection].replace([np.inf, -np.inf], np.nan)

    nona_columns = (train_data.isna().sum() + test_data.isna().sum() == 0)
    na_columns = nona_columns[~nona_columns].index
    nona_columns = nona_columns[nona_columns].index
    
    train_x = train_data[nona_columns].values
    test_x = test_data[nona_columns].values
    
    print(date) #, na_columns.to_list())
    

    # Standardization
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)


    # Truncation
    if not rank_y:
        mean = np.mean(train_y, axis=0)
        sd = np.std(train_y.astype(float), axis=0)
        train_y=np.array(train_y)
        cconst=2
        train_y[train_y > (mean + cconst * sd)] = mean + cconst * sd
        train_y[train_y < (mean - cconst * sd)] = mean - cconst * sd


    pred_y, model_info = prediction(train_x, train_y, test_x, method)

    if generate_modelinfo:
        drop_var = np.array(na_columns)
        
        if model_info is not None:
            coef = model_info['coef']
            pos = np.where(abs(coef)>(10**(-5)))[0]
            sel_var = np.array(nona_columns[pos])
            
            model_info['position'] = pos
            model_info['selected variables'] = sel_var
        else:
            model_info = {}
        
        model_info['dropped variables'] = drop_var
        model_info['all variables'] = np.array(x_var_selection)
        model_info['valid variables'] = np.array(nona_columns)
        model_infos[date] = model_info
    
    if generate_corr:
        daily_hist = daily['PCT_RET'].unstack().loc[daily.unstack().index < date]
    
        corr = daily_hist.corr()
        

        corr = corr[corr.index.get_level_values('SYMBOL').isin(test_data.index.get_level_values('SYMBOL').unique())]
        corr = corr.T
        corr = corr[corr.index.get_level_values('SYMBOL').isin(test_data.index.get_level_values('SYMBOL').unique())]
        corr = corr.T

        hist_corrs[date] = corr
    
    df.loc[date, 'pred_y'] = pred_y
    df.loc[date, 'pred_y_rank'] = df.loc[date, 'pred_y'].rank(ascending=1).values

    meta.loc[date, 'corr'] = df.loc[date, 'pred_y'].corr(df.loc[date, 'y'])
    meta.loc[date, 'corr_rank'] = df.loc[date, 'pred_y_rank'].corr(df.loc[date, 'y'].rank(ascending=1))
    


if generate_corr:
    np.save('../_data/hist_corr.npy', hist_corrs)
if generate_modelinfo:
    np.save('../_data/model_info{}.npy'.format(no), model_infos)

pred_df = df[['PCT_RET_t1', 'y', 'pred_y', 'pred_y_rank']].dropna(subset=['pred_y'], how='all', axis=0)
pred_df['IND_CODE'] = raw_df['IND_CODE'].groupby(level='SYMBOL').shift(1)
pred_df.to_csv('../_data/pred{}.csv'.format(no))
meta.to_csv('../_data/meta{}.csv'.format(no))

msg = """
no.: {}
method: {}
rank(x, y): {}
avg corr_rank: {}
avg corr_rank_rank: {}
vars: {}
""".format(no, method, (rank_x, rank_y), meta['corr'].mean(), meta['corr_rank'].mean(), x_var_selection)

print(msg)
import numpy as np
import pandas as pd

daily_monthly = False

pop_size = 700
gen = 10
ADV_filter = False


######################################## All variables excluding dummies
x_var_selection = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'BS_SH_OUT', 'CURR_ENTP_VAL', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP', 'MOV_AVG_30D', 'MOV_AVG_60D', 'MOV_AVG_200D', 'MOV_AVG_5D', 'MOV_AVG_10D', 'MOV_AVG_20D', 'MOV_AVG_40D', 'MOV_AVG_50D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'WACC', 'APPLIED_BETA', 'LT_DEBT_TO_TOT_EQY', 'NAV_FY1', 'DEBT_FY1', 'CPS_FY1', 'CAPEX_FY1', 'PROFIT_FY1', 'DEPRECIATION_FY1', 'EBIT_FY1', 'INCOME_FY1', 'OPP_FY1', 'PTP_FY1', 'NAV_FY0', 'DEBT_FY0', 'CPS_FY0', 'CAPEX_FY0', 'PROFIT_FY0', 'DEPRECIATION_FY0', 'EBIT_FY0', 'INCOME_FY0', 'OPP_FY0', 'PTP_FY0', 'NAV_FY2', 'DEBT_FY2', 'CPS_FY2', 'CAPEX_FY2', 'PROFIT_FY2', 'DEPRECIATION_FY2', 'EBIT_FY2', 'INCOME_FY2', 'OPP_FY2', 'PTP_FY2']

######################################## All variables including dummies excluding accounting
#x_var_selection = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'IND_CODE', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN']

######################################## All variables excluding dummies excluding accounting
#x_var_selection = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE_CHG', 'PCT_RET', 'TURNOVER', 'MKT_CAP', 'ISSUED_SHARE', 'EBITDA_EV', 'SAL_EV', 'PB', 'PE_T12MOB', 'PSALES', 'CAPEX_DEP_12M', 'BARRY_R', 'CURRENT_R', 'DEBT_EQUITY', 'GROSS_MARGIN', 'ROA', 'ROE', 'SAL_TA', 'CFY_FY0', 'EBP_MDN', 'EPSY_FY0', 'EPSY_FY1_AVG', 'EPSY_FY2_AVG', 'FY1_YLD', 'FY2_YLD', 'PE_FY0', 'TRL_YLD', 'RTN12_1M', 'RTN1260D', 'RTN21D', 'RTN252D', 'MA_CO_15_36W', 'REAL_VOL_1YD', 'SKEW_1YD', 'EXP_YLD', 'EPSY_T12MOB', 'CFY_IS', 'FCF_YLD', 'YOY_EPS_G', 'P_52WHI', 'P_52WLO', 'ROIC', 'ALTMAN', 'PAYOUT_OEPS', 'CHG_SHARES', 'CHG_DEBT', 'RNOA', 'CFROE_CF', 'CFROC_CF', 'SHORT_COV', 'MOM1M', 'MOM12M', 'INDMOM', 'MVEL1', 'MAXRET', 'CHMOM', 'RETVOL', 'DOLVOL', 'MOM6M', 'SP', 'TURN', 'NINCR', 'RD_MVE', 'AGR', 'STD_TURN', 'DY', 'INVEST', 'ILL', 'EP', 'CONVIND', 'CHINV', 'MOM36M', 'BETA', 'PS', 'ZEROTRADE', 'LGR', 'DEPR', 'BM', 'BETASQ', 'OPERPROF', 'BM_IA', 'LEV', 'BPS_FY0', 'BPS_FY1', 'BPS_FY2', 'CAPEX', 'CLOSE_FY0', 'CLOSE_FY1', 'CLOSE_FY2', 'COGS', 'COMEQ', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'DPS', 'DPS_FY0', 'DPS_FY1', 'DPS_FY2', 'EBITDA', 'EBITDA_FY0', 'EBITDA_FY1', 'EBITDA_FY2', 'EPS', 'EPS_FY0', 'EPS_FY1', 'EPS_FY2', 'EV', 'FCF', 'FCF_FY0', 'FCF_FY1', 'FCF_FY2', 'FY0_YLD', 'INC_BXO', 'INTEXP', 'INV', 'INV_CAP', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'NEWS_NEG_SENTIMENT_COUNT', 'NEWS_NEUTRAL_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT', 'NEWS_SENTIMENT_DAILY_AVG', 'NEWS_SENTIMENT_DAILY_MAX', 'NEWS_SENTIMENT_DAILY_MIN', 'NOA', 'OPCF', 'OPEXP', 'OPINC', 'PB_FY0', 'PB_FY1', 'PB_FY2', 'PCF', 'PCPS_FY0', 'PCPS_FY1', 'PCPS_FY2', 'PE_FY1', 'PE_FY2', 'PEG_FY0', 'PEG_FY1', 'PEG_FY2', 'PPEGT', 'RATING_FY0', 'RATING_FY1', 'RATING_FY2', 'RD_EXP', 'ROA_FY0', 'ROA_FY1', 'ROA_FY2', 'ROE_FY0', 'ROE_FY1', 'ROE_FY2', 'SALES', 'SALES_FY0', 'SALES_FY1', 'SALES_FY2', 'TOT_ASSET', 'TOT_DEBT', 'TOT_LIAB', 'TWITTER_NEG_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT', 'TWITTER_POS_SENTIMENT_COUNT', 'TWITTER_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_MAX', 'TWITTER_SENTIMENT_DAILY_MIN']

######################################## accounting variables
#x_var_selection = ['COGS', 'CUR_ASSET', 'CUR_LIAB', 'DEP', 'FCF', 'INTEXP', 'INV', 'ISS_COMSTOCK', 'LONGDEBT', 'NETINCOME', 'OPCF', 'OPEXP', 'OPINC', 'PPEGT', 'RD_EXP', 'SALES', 'BS_CASH_NEAR_CASH_ITEM', 'BS_ACCT_NOTE_RCV', 'BS_TOT_NON_CUR_ASSET', 'BS_NET_FIX_ASSET', 'BS_LT_INVEST', 'BS_ACCT_PAYABLE', 'BS_ST_BORROW', 'BS_TOT_EQY', 'CF_NET_INC', 'CF_CASH_FROM_INV_ACT', 'CF_DISP_FIX_ASSET', 'CF_CAP_EXPEND_PRPTY_ADD', 'CF_INCR_INVEST', 'CF_DECR_INVEST', 'CF_CASH_FROM_FNC_ACT', 'CF_DVD_PAID', 'CF_INCR_LT_BORROW', 'CF_REIMB_LT_BORROW', 'CF_INCR_CAP_STOCK', 'CF_DECR_CAP_STOCK', 'GROSS_PROFIT', 'IS_NET_NON_OPER_LOSS', 'PRETAX_INC', 'IS_INC_TAX_EXP']

######################################## price variables
#x_var_selection = ['PRE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']






from df import loading
monthly, daily, x_var_selection = loading(x_var_selection, ADV_filter = False)
if daily_monthly:
    data = daily[x_var_selection]
else:
    data = monthly[x_var_selection]
dates = monthly.index.get_level_values('DATE').drop_duplicates()
    
    
#std = monthly['PCT_RET'].std()
#monthly.loc[monthly['PCT_RET'] > (3 * std), 'PCT_RET'] = 3 * std
#monthly.loc[monthly['PCT_RET'] < (-3 * std), 'PCT_RET'] = -3 * std


#from sklearn.preprocessing import StandardScaler
#
#stdscaler = StandardScaler().fit(monthly[x_var_selection])
#monthly[x_var_selection] = stdscaler.transform(monthly[x_var_selection])


d = {'ARG{}'.format(i): v for i, v in enumerate(x_var_selection)}
xs = [data[col] for col in x_var_selection]





def return_from_score(temp_data):   
    corrs = temp_data.groupby(mth).apply(lambda x: x['x'].corr(x['y'])).fillna(0)
    temp_data_rank = temp_data.groupby(mth).rank(ascending=True)
    corr_ranks = temp_data_rank.groupby(mth).apply(lambda x: x['x'].corr(x['y'])).fillna(0)
    
#    sign = np.sign(corr.shift(-3))
    sign = 1
    temp_data['x'] = temp_data['x'] * sign
    
    monthly_profolio = temp_data.dropna().groupby('DATE', group_keys=False).apply(lambda x: x.sort_values('x').tail(10))
    rets = temp_data['y'][temp_data.index.isin(monthly_profolio.index)].groupby('DATE', group_keys=False).mean().reindex(index=dates).fillna(0)
    
    return rets, corrs, corr_ranks


from exprstr import compute_metrics

def expr_print(individual):
    func = toolbox.compile(expr=individual)
    expr = sympy.simplify(stringify_for_sympy(individual))
    
    x1 = func(*xs)
    y = monthly['PCT_RET']
    x = x1[x1.index.isin(y.index)].groupby(symbol).shift(1)
    temp_data = pd.concat([x, y], axis=1)
    temp_data.columns = ['x', 'y']
    
    rets, corrs, corr_ranks = return_from_score(temp_data)
    
    metrics = compute_metrics(rets)
    
    print('{:.4f}, {:.4f}, {}, {}'.format(gmean(rets), rets.mean(), len(individual), expr))
    return rets, corrs, corr_ranks, metrics

rets = []
corrs = []
corr_ranks = []
metrics = {}

def r():
    global rets, corrs, corr_ranks, metrics
    for i in range(9, -1, -1):
        rets, corrs, corr_ranks, metrics = expr_print(hof[i])
    
    import matplotlib.pyplot as plt
    rets.hist()
    plt.show()
    corrs.hist()
    plt.show()

    rets.plot()
    plt.show()
    
    print('corr', corrs.mean())
    print('corr_rank', corr_ranks.mean())
    
    from pprint import pprint
    pprint(metrics)





import operator
import random
import time
import sympy

from deap import gp
from deap import tools
from deap import algorithms
from deap import base
from deap import creator

import warnings
warnings.simplefilter(action='ignore')

from exprstr import stringify_for_sympy

mth_symbol = [pd.Grouper(freq='M', level='DATE'), pd.Grouper(level='SYMBOL')]
mth = [pd.Grouper(freq='M', level='DATE'), ]
symbol = [pd.Grouper(level='SYMBOL'), ]


# Define new functions
def protectedDiv(left, right):
    try:
        result = left / right
        if type(left) is series and type(right) is series:
            return scaler(result.replace([np.inf, -np.inf], np.nan))
        else:
            return result
    except ZeroDivisionError:
        return 1

def iden(int1):
    return int1

def int_div(int1, s1):
    return protectedDiv(int1, s1)

def lag(s1, int1):
    return s1.groupby(symbol).shift(int1).sort_index(level=['DATE', 'SYMBOL'])

def mean(s1, int1):
    return s1.groupby(symbol, group_keys=False).rolling(int1).mean()

def std(s1, int1):
    sd = s1.groupby(symbol, group_keys=False).rolling(int1).std()
    return sd

def gtz(s1):
    gt = (s1 > 0).astype(int)
    return scaler(gt) 

def ltz(s1):
    lt = (s1 < 0).astype(int)
    return scaler(lt) 

series = pd.core.series.Series
#scaler = series

class scaler(pd.core.series.Series):
    pass

def s_add(s1, s2):
    return scaler(s1 + s2)

def s_sub(s1, s2):
    return scaler(s1 - s2)


def gmean(s1):
    from scipy import stats
    s1 = s1.fillna(0)
    return (stats.gmean(s1 / 100 + 1) - 1) * 100







pset = gp.PrimitiveSetTyped("MAIN", [series] * (len(x_var_selection)) , scaler)

#pset.addPrimitive(int_div, [int, series], series)
pset.addPrimitive(lag, [series, int], series)
pset.addPrimitive(mean, [series, int], series)
pset.addPrimitive(std, [series, int], series)
#pset.addPrimitive(gtz, [series], scaler)
#pset.addPrimitive(ltz, [series], scaler)

pset.addPrimitive(operator.add, [series, series], series)
pset.addPrimitive(operator.sub, [series, series], series)
pset.addPrimitive(operator.mul, [series, scaler], series)
pset.addPrimitive(protectedDiv, [series, series], scaler)

pset.addPrimitive(s_add, [scaler, scaler], scaler)
pset.addPrimitive(s_sub, [scaler, scaler], scaler)

pset.addPrimitive(iden, [int], int)
#pset.addPrimitive(iden, [series], series)
#pset.addPrimitive(iden, [scaler], scaler)

#for i in range(0, 5):
#   pset.addTerminal(i, int)

s1 = scaler(pd.Series(index = monthly['CLOSE'].dropna().index, data=1))
pset.addTerminal(s1, scaler, name='scaler')

#pset.addEphemeralConstant("1_{}".format(random.random()), lambda: 1)
pset.addEphemeralConstant("rand15_{}".format(time.time()), lambda: random.randint(1, 20), int)
#pset.addEphemeralConstant("rand01_{}".format(random.random()), lambda: random.random())

pset.renameArguments(**d)


creator.create("FitnessMin", base.Fitness, weights=(1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=2, type_=scaler) #genHalfAndHalf
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)  

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


score_dict = {}
x1 = {}
temp_data = []

def evalSymbReg(individual):
    global score_dict, x1, temp_data
    
    func = toolbox.compile(expr=individual)
    expr = sympy.simplify(stringify_for_sympy(individual))
    

    if expr in score_dict:
        ret = score_dict[expr]
#        print( expr )
    else:
        x1 = func(*xs)
        
        try:
            if type(x1) is not series and type(x1) is not scaler:
                ret = 0
                corr = 0
                corr_rank = 0
            else:
                if type(x1) is series:
                    t = 'series'
                else:
                    t = 'scaler'
                
                y = monthly['PCT_RET']
                x = x1[x1.index.isin(y.index)].groupby(symbol).shift(1)
                
                temp_data = pd.concat([x, y], axis=1)
                temp_data.columns = ['x', 'y']
                
                rets, corrs, corr_ranks = return_from_score(temp_data)
#                ret = rets.mean()
                ret = gmean(rets)
                score_dict[expr] = ret
                
            print('{:.4f}, {}, {}'.format(ret, t, str(expr)))
        except Exception as e:
            print(expr)
            raise e
            
    
    return ret, len(individual)




toolbox.register("evaluate", evalSymbReg) # xs_daily, y_daily, y_monthly)


def main():
    
    random.seed(318)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, gen, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    
    expr = hof[0]
    nodes, edges, labels = gp.graph(expr)
    
    import matplotlib.pyplot as plt
    import networkx as nx
    
#    import pydotplus
    from networkx.drawing.nx_pydot import graphviz_layout
    
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")
    
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()    
    
    s = stringify_for_sympy(expr)
    
    print( sympy.simplify(s) )
    print('#'*80)
    print( sympy.expand(s) )
    
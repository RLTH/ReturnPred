def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    import copy
    prim = copy.copy(prim)
    #prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        's_sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'protectedDiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'int_div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'mul': lambda *args_: "Mul({},{})".format(*args_),
        'add': lambda *args_: "Add({},{})".format(*args_),
        's_add': lambda *args_: "Add({},{})".format(*args_),
        'neg': lambda *args_: "-({})".format(*args_),
        'iden': lambda *args_: "{}".format(*args_),
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)

def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


import pandas as pd
from scipy import stats
import statsmodels.api as sm

rf = pd.read_csv('../../_data/mkt.csv')
rf = rf.set_index('DATE')
rf.index = pd.to_datetime(rf.index)

def compute_metrics(rets):
    
    # For CAPM model, y = portfolio risk-premium, x = market risk-premium
    series = rets
    series_mkt = rf['HSI_PCT_RET']
    series_rf = rf['RF']        
    y = series - series_rf
    x = series_mkt - series_rf
    series_1base = series / 100 + 1
    series_cum = series_1base.cumprod()
    win_market = (series - series_mkt).dropna() > 0
    
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
    
    return metrics























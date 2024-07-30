import pandas as pd
# from utils import validate
from globals import *


def rolling_kdj(df, action_nextday = False) -> pd.Series:
    rolling_actions = df.apply(kdj, axis=1) # 按行
    
    if action_nextday: 
        rolling_actions = rolling_actions.shift(1).fillna(0).astype("int")

    return rolling_actions

# 用rolling_action针对每行
def kdj(row):
    if row.kdj_vr > row.kdjd:
        action = 1  # buy
    elif row.kdj_vr < row.kdjd:
        action = -1 # sell
    else:
        action = 0  # hold
    return action

def ma_return(row):
    return 1 if row.ma_r > EPS +1 else -1 if row.ma_r < 1- EPS else 0



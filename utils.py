import pandas as pd
import stockstats
import requests
import matplotlib as mpl
from datetime import datetime, timedelta

WINDOW_SIZE = 20
# 检查df的合法性
def validate(d, required, typeof=[pd.DataFrame, stockstats.StockDataFrame], index_type=None):    
    ''' validate whether or not:
    1. The type of d in list `typeof`
    2. d should not be empty
    3. d at least contain columns desinated by `required`
    4. The type of d.index in list `index_type`
    '''
    if d.empty:
        raise ValueError("The DataFrame must not be empty.")

    if not any(isinstance(d,i) for i in typeof):   # 必须是其中之一
        raise TypeError(f"Type must be either of {typeof}")

    if not all(f in d.columns for f in required): # 所有字段都必须存在
        raise ValueError(f"The DataFrame must contain columns:{required}")

    if index_type is not None:
        if not any(isinstance(d.index, i) for i in index_type): # index的类型必须在限定范围里
            raise ValueError(f"Index type must be one of {index_type}")


def attach_grey(sdf, axes):
    
    validate(sdf, required=['action','close'])
    # also requires `date` as the index
    sdf['norm_close'] = sdf.close/sdf.close.iloc[0]
    sdf['last_action'] = sdf.action.shift(1).fillna(0)    # compared with yesterday
    action_days = sdf[sdf.action != sdf.last_action] 
    action_days = action_days.reset_index() # 将index恢复为column
    action_days['next_day'] = action_days.shift(-1).date
    action_days['next_day'] = action_days['next_day'].fillna(sdf.index[-1]) # 用d最后一行的date补齐最后一行的空值
    short_days = action_days[action_days.action == -1][['next_day']] # 一加上这后缀，sdf就自己把date设为index

    def plot_range(x, x_axis, floor, ceiling):
        ax.fill_between(x_axis, floor, ceiling, (x.date <= x_axis) & (x_axis< x.next_day), color = "k", alpha = 0.05)
    
    if isinstance(axes, mpl.axes._axes.Axes):   # if it is a single instance other than an array of them.
        axes = [axes] 
        
    short_days = short_days.reset_index()   # index恢复成`date`做空的期间 (date, next_day)
    for ax in axes:
        floor, ceiling = ax.get_ylim()
        short_days.apply(plot_range, args=(sdf.index, floor, ceiling), axis=1) 

    return short_days

import numpy as np
def super_smoother(data:np.array, length) -> np.array:
    a1 = np.exp(-1.414 * 3.14159 / (0.5 * length))
    b1 = 2 * a1 * np.cos(1.414 * 180 / (0.5 * length))
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # filt = [0] * len(data)
    filt = np.zeros_like(data)
    filt[0] = data[0]
    filt[1] = c1 * (data[0] + data[1]) / 2 + (c2+c3) * filt[0]
    for i in range(2, len(data)):
        filt[i] = c1 * (data[i] + data[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
        
    return filt

def evaluate(sdf)->pd.DataFrame:
    validate(sdf, required=['close', 'action'])
    
    # 有些策略有可能会留下0，这里不考虑hold的情况
    sdf['action'] = sdf['action'].replace(0,np.nan).ffill()  # hold的内涵：前面为buy时我继续buy，前面sell时我继续sell
    sdf['pctchg'] = sdf.close.pct_change().fillna(0)
    
    ## 计算B&H
    underline = sdf.close/sdf.close.iloc[0]            # 将close归1
    ## 1x
    long1x = sdf.pctchg[sdf.action == 1] + 1            # 取移动平均收益率>1的部分
    short1x = (sdf.pctchg[sdf.action == -1] + 1)**(-1)  # 取移动平均收益率<1的部分，做空
    flat1x = (sdf.pctchg[sdf.action == -1] + 1)**(0)      # 取移动平均收益率<1的部分，不做空，即全赋值为1
    do_short = pd.concat([long1x,short1x], axis=0).sort_index().cumprod()   # 拼接两部分，做空
    no_short = pd.concat([long1x,flat1x], axis=0).sort_index().cumprod()      # 拼接两部分，不做空
    
    ## 另外一种做法
    # do_short = sdf.apply(lambda x: (x.pctchg+1)**(x.action), axis=1).cumprod()  # 作多时change**(1)，做空时，change**(-1)
    # no_short = sdf.apply(lambda x: (x.pctchg+1)**(1 if x.action == 1 else 0), axis=1).cumprod() # 没有做空的时候，相比上一天没变化，所以是 **(0) -> 1
    # ETF2x_short = sdf.apply(lambda x: (x.pctchg*2+1)**(x.action), axis=1).cumprod()  # 作多时change**(1)，做空时，change**(-1)
    # ETF2x_no_short = sdf.apply(lambda x: (x.pctchg*2+1)**(1 if x.action == 1 else 0), axis=1).cumprod() # 没有做空的时候，相比上一天没变化，所以是 **(0) -> 1

    ## evaluate.py的做法
    # d['change_wo_short'] = d['change_w_short'] = d.close/d.close.shift(1) # 只做多情况下与上一天的变化比例，会在第一行留下nan
    # d.loc[d.action == -1,'change_wo_short'] = 1     # 不做空的日子，与上一天相比没变化，等同于 **=0
    # d.loc[d.action == -1,'change_w_short'] **= -1   # 有做空的日子，与上一天的比率取倒数

    ## 2x
    long2x = 2*sdf.pctchg[sdf.action == 1] + 1              # 取移动平均收益率>1的部分，2倍做多
    short2x = (2*sdf.pctchg[sdf.action == -1] + 1)**(-1)    # 取移动平均收益率<1的部分，2倍做空
    flat2x = (2*sdf.pctchg[sdf.action == -1] + 1)**(0)      # 取移动平均收益率<1的部分，不做空，即全赋值为1
    ETF2x_short = pd.concat([long2x,short2x], axis=0).sort_index().cumprod()    # 拼接两部分，2倍做空
    ETF2x_no_short = pd.concat([long2x,flat2x], axis=0).sort_index().cumprod()  # 拼接两部分，2倍不做空

    df = pd.DataFrame({"B&H": underline,
                    "1x_short":do_short, 
                    "1x_no_short":no_short,
                    "2x_short":ETF2x_short,
                    "2x_no_short":ETF2x_no_short,
                    "action":sdf.action})
    ## 输出最后一行作为观察
    print(df.tail(1))
    
    return df 

def llt_smoother(prices: pd.Series, alpha: float = 2/(60+1)) -> pd.Series:

    llt = pd.Series(index=prices.index, dtype='float64')

    # 需要至少两个价格点来计算LLT
    if len(prices) < 2:
        return prices

    # 初始化LLT的前两个值
    llt.iloc[0] = prices.iloc[0]
    llt.iloc[1] = prices.iloc[1]

    # 使用给定的公式计算接下来的LLT值
    for t in range(2, len(prices)):
        llt.iloc[t] = ((alpha - alpha**2 / 4) * prices.iloc[t] +
                  (alpha**2 / 2) * prices.iloc[t-1] -
                  (alpha - 3 * alpha**2 / 4) * prices.iloc[t-2] +
                  2 * (1 - alpha) * llt.iloc[t-1] -
                  (1 - alpha)**2 * llt.iloc[t-2])
    
    return llt

def kalman_filter(observations:pd.Series, damping_factor=0.9, initial_value=0)->pd.Series:
    # 初始化
    estimated_value = initial_value
    estimated_error = 1.0
    
    result = []
    for observation in observations:
        # 预测
        predicted_value = estimated_value
        predicted_error = estimated_error + (1 - damping_factor)
        
        # 更新
        kalman_gain = predicted_error / (predicted_error + 1)
        estimated_value = predicted_value + kalman_gain * (observation - predicted_value)
        estimated_error = (1 - kalman_gain) * predicted_error
        
        result.append(estimated_value)
    
    return pd.Series(result, index=observations.index)

def sz50constituents():
    # 文件URL
    # hs300
    # url = "https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/static/html/csindex/public/uploads/file/autofile/closeweight/000300closeweight.xls"
    # sz50
    url = "https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/static/html/csindex/public/uploads/file/autofile/closeweight/000016closeweight.xls"
    columns = ['成份券代码Constituent Code', '成份券名称Constituent Name', '权重(%)weight']
    new_columns = ['code', 'name', 'weight']
    response = requests.get(url)
    sz50_constituents = pd.read_excel(response.content,header=0, usecols=columns)
    sz50_constituents.columns = new_columns
    # sz50_constituents = sz50_constituents.sort_values('code').set_index('code')
    return sz50_constituents

import akshare as ak
from stockstats import StockDataFrame
def get_stock(code:str, days:int=None)->pd.DataFrame:
    end_date = datetime.today().strftime("%Y-%m-%d")
    days = 365 if days is None else days
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        
    index_code = "sh"+str(code)
    df = ak.stock_zh_a_daily(index_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('date').set_index('date')
    return StockDataFrame(df) 



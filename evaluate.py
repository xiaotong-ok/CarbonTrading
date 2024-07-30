import pandas as pd
import numpy as np
from stockstats import StockDataFrame
# from utils import validate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_context("notebook")

class Evaluator():
    def __init__(self, sdf:StockDataFrame):
        # validate(sdf, required=['close', 'action'])
        self.sdf = sdf
        self.result = self.evaluate()   # 业绩序列
        # 做空的日期
    
    def evaluate(self) -> pd.DataFrame:
        sdf = self.sdf.copy()
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
        self.result = df
        return self.result
    
    def collect_shorts(self):
        sdf = self.result.copy()
        sdf['norm_close'] = sdf['B&H']/sdf['B&H'].iloc[0]
        sdf['last_action'] = sdf.action.shift(1).fillna(0)    # compared with yesterday
        action_days = sdf[sdf.action != sdf.last_action] 
        self.action_days = action_days
        action_days = action_days.reset_index() # 将index恢复为column
        action_days['next_day'] = action_days.shift(-1).date
        action_days['next_day'] = action_days['next_day'].fillna(sdf.index[-1]) # 用d最后一行的date补齐最后一行的空值
        self.short_days = action_days[action_days.action == -1][['date','next_day']]
        
    def attach_grey(self, axes):

        def plot_range(x, x_axis, floor, ceiling):
            ax.fill_between(x_axis, floor, ceiling, (x.date <= x_axis) & (x_axis< x.next_day), color = "k", alpha = 0.05)
        
        if isinstance(axes, mpl.axes._axes.Axes):   # if it is a single instance other than an array of them.
            axes = [axes] 
        
        short_days = self.short_days.copy()       
        short_days = short_days.reset_index(drop=True)   # index恢复成`date`，plot_range中要用到

        for ax in axes:
            floor, ceiling = ax.get_ylim()
            short_days.apply(plot_range, args=(self.sdf.index, floor, ceiling), axis=1) 
            ax.set_xticks(pd.concat([short_days.date, short_days.next_day], axis=0).sort_index())
            ax.tick_params(axis='x', labelsize=6)
    
    def attach_grey2(self, axes):
        def plot_range(x, x_axis, floor, ceiling):
            if len(x) < 2:
                return 
            day1 = x.index[0]
            day2 = x.index[1]
            action1 = x[0]
            print(day1, day2, action1)
            ax.fill_between(x_axis, floor, ceiling, 
                            (day1 <= x_axis) & (x_axis< day2), 
                            color = "k" if action1 == -1 else "w", 
                            alpha = 0.05
                            )
        def plot_range2(x, x_axis, floor, ceiling):
            pass
        
        if isinstance(axes, mpl.axes._axes.Axes):   # if it is a single instance other than an array of them.
            axes = [axes] 
        
        for ax in axes:
            floor, ceiling = ax.get_ylim()
            self.action_days.action.dropna().rolling(2, min_periods=2).apply(plot_range2, args=(self.sdf.index, floor, ceiling)) 
            ax.set_xticks(self.action_days.index)
            ax.tick_params(axis='x', labelsize=6) 

    def plot(self, **kwargs):
        fig, axes = plt.subplots(2,1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, dpi=120)
        self.result[['1x_short', '1x_no_short','2x_short', '2x_no_short']].plot(ax=axes[0], colormap="viridis",**kwargs)
        self.result["B&H"].plot(ax=axes[1], rot=90, **kwargs)
        self.collect_shorts()
        self.attach_grey(axes)
        return fig 
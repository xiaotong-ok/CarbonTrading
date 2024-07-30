# CAUTION: globals variables could be imported by other modules widely, so don't do it wildly. 
# Any modification may lead to unpredictable consequence!

import os
MAIN_PATH = os.path.dirname(__file__)  # 当前文件所在的目录
os.chdir(MAIN_PATH)
print(f"Working in {MAIN_PATH}")

#%%
from collections import namedtuple
Action = namedtuple('Action', ['buy', 'hold', 'sell'])
action = Action(1,0,-1)     # however, gym.spaces.Discrete(3) allows [0,1,2] only.
Flex = namedtuple('Flex', ['peak', 'waist', 'bottom'])
flex = Flex(-1,0,1)
#%% 
WINDOW_SIZE = 20
LookBack = 20  # same as window_size
LookAhead = 5  # same as horizon
DATE_FORMAT = "%Y-%m-%d"
REWARD = ['landmark','buy_reward','sell_reward','hold_reward']
WEEKDAY = ['dayofweek']   # 注意这个是字符串，在和其他list相连的时候需要list + [WEEKDAY]，主要是用在对df[WEEKDAY]直接赋值
Symbol = '000001'
Period = 'daily'
StartDay = '20200101'
EndDay = '20240721'
MODEL_PATH  = 'mlp000001.pkl'

#%% prefix
MARKET_PREFIX = "mkt_"
ETF_PREFIX = "etf_"
SECTOR_PREFIX = "sct_"
#%% column definitions
TD = ['ticker', 'date']
TDT = ['ticker', 'date','time']
OCLHVA = ['open',  'high', 'low', 'close', 'volume', 'amount']
OCLHVA_HELPER = ['preclose','turn','pctChg']
Normed_OCLHVA = [x+"_" for x in OCLHVA]     # normalized ohlcva
MKT_OCLHVA = OCLHVA 
mkt_oclhva_normed = [MARKET_PREFIX+x for x in Normed_OCLHVA]

# indicators collection # "MFI","EMV","VR","PSY","OBV" are volume-concerned indicators
# https://github.com/jealous/stockstats
indicators = ['close_5_ema', 'close_10_ema','close_20_ema',"rsi_6", "rsi_12"]
# indicators = ['kdjk', 'kdjd', 'kdjj']
# indicators += ['boll','boll_lb','boll_ub']
# indicators += ["rsi_6", "rsi_12", "rsi_24"]
# indicators += ["macds","macdh","atr","vr","adx"] 
# indicators += ['close_5_ema', 'close_10_ema','close_20_ema','close_50_ema','close_100_ema']     # 有很多策略要用到很长周期的均线，若基于20日窗口就没法计算
mkt_indicators = [MARKET_PREFIX+x for x in indicators]
sct_indicators = [SECTOR_PREFIX+x for x in indicators]
# for reflex, refer to https://zhuanlan.zhihu.com/p/557480350
#%%
TS_TOKEN = "72d1e47c3b0728a26bfc4a9f54132b195890fa843815f896708515f1"

#%% data source mapping dictionary, used by pd.rename()
TUSHARE_MAPPING = {"ts_code":"ticker","trade_date":"date","vol":"volume"}
BAOSTOCK_MAPPING = {"code":"ticker"}

#%%
# # 上证50ETF成分股
# 上证50指数依据样本稳定性和动态跟踪相结合的原则，每半年调整一次成份股，调整时间与上证180指数一致。特殊情况时也可能对样本进行临时调整。
# 每次调整的比例一般情况不超过10％。样本调整设置缓冲区，排名在40名之前的新样本优先进入，排名在60名之前的老样本优先保留

SH50Index = "000016" # 上证50指数代码
SH50ETF = "510050" # 上证50ETF代码

summary = 'evaluated.csv'   # 统计文件名称
# 生成的etf信号，
# 结构与股票的测试完全一致:['ticker',	'date',	'close'	'action',	'reward',	'change_wo_short',	'change_w_short']
etf_action = 'etf_action.csv'   
test_result = "Test"    # 测试结果存放目录

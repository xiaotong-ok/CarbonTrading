from torch.utils.data.dataset import Dataset
import torch.utils.data.dataloader as DataLoader
import pandas as pd
import numpy as np
import torch 
from globals import LookBack, indicators, WEEKDAY, OCLHVA
from functools import partial
from stockstats import StockDataFrame
from transformers import BertModel, BertTokenizer
import re, json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import validate

class finDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.data = prepare(data)
        self.feature = self.data.iloc[:,:-1]
        self.label = self.data['reward']
        
    def __len__(self):
        return len(self.data) - LookBack + 1
    
    def __getitem__(self, index):
        start_index = index
        end_index = index + LookBack
        feature = torch.tensor(self.feature.iloc[start_index:end_index].values, dtype=torch.float32)
        label = torch.tensor(self.label.iloc[end_index-1], dtype=torch.float32)  # Use the last value in the window as the label
        return feature, label
    
    @property
    def features(self, index):
        start_index = index
        end_index = index + LookBack
        return torch.tensor(self.feature.iloc[start_index:end_index].values, dtype=torch.float32)
    
    @property
    def labels(self, index):
        start_index = index
        end_index = index + LookBack
        return torch.tensor(self.label.iloc[end_index-1], dtype=torch.float32)  # Use the last value in the window as the label

    @property
    def size(self):
        return self.feature.shape[1]
    
def data_generator(df, test_size=0.3):
    df = prepare(df)
    # train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    cut_line = round(len(df)*0.7)
    train_df = df.iloc[:cut_line]
    test_df = df.iloc[cut_line:]
    train_dataset = finDataset(train_df)
    test_dataset = finDataset(test_df)
    train_dataloader, test_dataloader = \
        DataLoader.DataLoader(train_dataset, batch_size=32), \
        DataLoader.DataLoader(test_dataset, batch_size=32)
    data_size = (df.shape[1]-1) * LookBack
    return train_df, test_df, train_dataloader, test_dataloader, data_size
        
def prepare(sdf):
    ''' Add indicators defined in globals.py to StockDataFrame `sdf`
    '''
    # validate(sdf, ['open','close','low','high','volume']) 
    # sdf = sdf[OCLHVA]

    sdf = add_indicators(sdf)
    sdf = sdf.dropna()
    feature_columns = sdf.columns
    reward5(sdf)
    sdf = sdf.dropna()
    
    scaler = MinMaxScaler()
    sdf[feature_columns] = pd.DataFrame(scaler.fit_transform(sdf[feature_columns]), columns=feature_columns, index=sdf.index)
    
    return sdf

''' Specification
@param df:pd.DataFrame
@param action_nextday: bool # 第二天才action，效果差了很多很多
@return pd.Series of action:int \in [-1, 0, 1]
'''

def attach_reward(df, look_ahead):
    ''' attach rewards to corresponding sliding windows each.
    '''
    df['reward'] = df["close"].rolling(window=look_ahead+1, min_periods=look_ahead+1).apply(reward)
    df['reward'] = df['reward'].shift(-look_ahead)  # 计算好以后往上移，与close并列


reward5 = partial(attach_reward, look_ahead=5)
reward10 = partial(attach_reward, look_ahead=10)
reward20 = partial(attach_reward, look_ahead=20)
    
def reward(landscape:pd.Series) -> float:
    ''' determine the rank a price in a prices set -- a rudimentary version
    to be more delicate, try quantile
    '''
    return landscape.rank(pct=True,ascending=False).iloc[0]

def add_indicators(df):
    sdf = StockDataFrame(df.copy()) 
    # 可以改为sdf[WEEKDAY], global部分要改为WEEKDAY='weekday'
    df['weekday'] = pd.to_datetime(sdf.index).dayofweek
    # kdj 要单独计算，stockstat调用不出来
    df[indicators] = sdf[indicators]
    df['kdjk'], df['kdjd'],df['kdjj'] = calculate_kdj(df)
    df = df.dropna()
    return df

def calculate_kdj(data, period=9, k_period=3, d_period=3):
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    
    rsv = (data['close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(0) 
    
    k = rsv.ewm(com=(k_period-1), adjust=False).mean()
    d = k.ewm(com=(d_period-1), adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def process_price(price_df):
    price_df = price_df.drop(columns =['name'])
    price_df['low'] = pd.to_numeric(price_df['low'], errors='coerce')
    price_df['high'] = pd.to_numeric(price_df['high'], errors='coerce')  
    price_df['date'] = shift_datetime(price_df['date'],shift_from='%Y%m%d', shift_to='%Y-%m-%d')
    price_df = price_df.sort_values(by='date', ascending=True).set_index('date')
    return price_df

def process_event_for_cl(event_df):
    '''
    Processes an event DataFrame by shifting dates, removing duplicates, 
    and formatting event types and triggers

    '''
    df = event_df.copy()
    df['date'] = df['date'].shift(-1)
    df = df.dropna()
    df = df[df['events']!='[]']
    first_col = df.pop('date')
    df.insert(0, 'date', first_col)
    
    df = event_chain(df)
    df = argument_process(df)
    
    df['type'] = df['type'].astype(str)
    df['type'] = df['type'].apply(lambda x: convert(x))
    df['trigger'] = df['trigger'].astype(str)
    df['trigger'] = df['trigger'].apply(lambda x: convert(x))
    df['arguments'] = df['arguments'].astype(str)
    df['arguments'] = df['arguments'].apply(lambda x: convert(x))
    
    df['date'] = shift_datetime(df['date'], shift_from='%Y-%m-%d', shift_to='%Y-%m-%d')
    # df = df.set_index('date')
    # df = df.groupby('date')['type','trigger','arguments'].transform(lambda x: ','.join(x)).reset_index().drop_duplicates()
    df = df.sort_values(by='date', ascending=True).set_index('date')
    return df

def process_event(event_df):
    '''
    Processes an event DataFrame by shifting dates, removing duplicates, 
    and formatting event types and triggers

    '''
    df = event_df.copy()
    df['date'] = df['date'].shift(-1)
    df = df.dropna()
    df = df[df['events']!='[]']
    first_col = df.pop('date')
    df.insert(0, 'date', first_col)
    
    df = event_chain(df)
    df = argument_process(df)
    
    df['type'] = df['type'].astype(str)
    df['type'] = df['type'].apply(lambda x: convert(x))
    df['trigger'] = df['trigger'].astype(str)
    df['trigger'] = df['trigger'].apply(lambda x: convert(x))
    df['arguments'] = df['arguments'].astype(str)
    df['arguments'] = df['arguments'].apply(lambda x: convert(x))
    
    df['date'] = shift_datetime(df['date'], shift_from='%Y-%m-%d', shift_to='%Y-%m-%d')
    df = df.set_index('date')
    df = df.groupby('date')['type','trigger','arguments'].transform(lambda x: ','.join(x)).reset_index().drop_duplicates()
    df = df.sort_values(by='date', ascending=True).set_index('date')
    return df

def argument_process(df):
    '''
    Process the 'arguments' column in the DataFrame, extracting the 'mention' from each element and storing it in a list.
    '''
    mention_list = []
    for i in df['arguments']:
        m_list=[]
        # print(i)
        for m in i[0]:
            # print(m)
            m_list.append(m['mention'])
        mention_list.append(m_list)
    df['arguments'] = mention_list

    return df


def event_chain(df):
    '''
    Integrate event into a sequence, and decompose into three parts
    '''
    type_list = []
    trigger_list = []
    argument_list = []
    for index, event in enumerate(df.iterrows()):
        # print(event[1]['events'])
        data_str = event[1]['events'].replace("'", '"')
        data = json.loads(data_str)
        # print(data)
        type_list.append([item['type'] for item in data])
        trigger_list.append([item['trigger'] for item in data])
        argument_list.append([item['arguments'] for item in data])
        
    df['type'] = type_list
    df['trigger'] = trigger_list
    df['arguments'] = argument_list
    return df 

def convert(string):
    '''Extracts content within square brackets from a given string.
    '''
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, string)
    if match:
        result = match.group(1).replace("'", '')
        return result
    else:
        print('未找到')

def add_event_embedding_sum(events, price, text):
    '''embed event, sum up the embedding and concatenate it with price
    '''
    def to_token(tokenizer, input):
        token = tokenizer(input)['input_ids']
        return sum(token)/len(token)

    events = process_event(events)
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese',
                trust_remote_code=True, local_files_only=False)
    events['type'] = events['type'].apply(lambda x: to_token(tokenizer, x))
    events['trigger'] = events['trigger'].apply(lambda x: to_token(tokenizer, x))
    events['arguments'] = events['arguments'].apply(lambda x: to_token(tokenizer, x))
    con = events[text].merge(price, left_index=True, right_index=True, suffixes=('_df1', '_df2'))
    con = con.dropna()
    return con

def add_event_embedding(data):
    data['embedding'] = data['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    embeddings_array = np.vstack(data['embedding'].values)

    for i in range(embeddings_array.shape[1]):
        data['column' + str(i)] = 0
    data.iloc[:, -embeddings_array.shape[1]:] = embeddings_array
    return data

    
    
def get_embeddings(tokenizer, model, text):
    '''
    Get the embeddings for the given text using the provided tokenizer and model.
    '''
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    
def shift_datetime(date, shift_from='%Y%m%d', shift_to='%Y%m%d'):
    date = pd.to_datetime(date, format=shift_from)
    date = date.dt.strftime(shift_to)
    return date
    
def prepare_for_cl(price, events, text=['type']):
    '''process price and events for constrative learning'''
    price = process_price(price)
    reward5(price)
    price = price.dropna()
    events = process_event_for_cl(events)
    df = pd.merge(events, price, on='date', how='left').dropna()
    
    df['total'] = df[text[0]]  
    for i in text[1:]:
        df['total'] += df[i]
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    df['token'] = df['total'].apply(lambda x: tokenizer(x)['input_ids'])
    df['embedding'] = df['total'].apply(lambda x: get_embeddings(tokenizer, model, x))
    data = df.copy()
    data = add_event_embedding(data)
    return data
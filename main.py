import pandas as pd
import akshare as ak
from sdl_strategy import train_mlp, test_mlp
from data import Data
from globals import *
from preprocess import *
from evaluate import *
import matplotlib.pyplot as plt
from preprocess import data_generator

def main(action_nextday = False, event_joined=True, retrain = True, eps=0.5, text=['type'], epoches=20):
    # price = Data(Symbol, Period, StartDay, EndDay).get_data()
    price = pd.read_csv('carbon_price.csv')
    events = pd.read_csv('event_extraction.csv')
    price = process_price(price)
    if event_joined:
        data = add_event_embedding_sum(events, price, text)
    else:
        data = price
    train_df, test_df, train_dataloader, test_dataloader, input_size = data_generator(data)
    test_result = test_mlp(train_df, test_df, train_dataloader, test_dataloader, input_size, epoches=epoches, eps=eps, action_nextday = action_nextday, retrain = retrain)
    # Evaluator(test_result).plot()
    # plt.show()
    return test_result
    
    

if __name__ == '__main__':
    main()
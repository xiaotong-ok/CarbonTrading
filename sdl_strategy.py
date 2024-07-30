import os
# os.chdir('/data/JupyterLab/xiaotong/strategies')
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from globals import MODEL_PATH, MODEL_PATH, LookBack
from preprocess import finDataset, reward, reward5, reward10, reward20
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import train_test_split

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)
        x = self.fc3(x)
        return x   
   

def train_mlp(train_dataloader, input_size, hidden_size, output_classes, epoches):    
    model = MLPModel(input_size, hidden_size, output_classes)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    acc_list = []
    loss_list = []
    for epoch in range(epoches):
        for i, item in enumerate(train_dataloader):
            feature, labels = item
            model.zero_grad()
            outputs = model(feature)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        print('Epoch [{}/{}], Loss: {}'.format(epoch+1, 100, loss.mean()))
    torch.save(model, MODEL_PATH)
    return model
            
            

def test_mlp(train_df, test_df, train_dataloader, test_dataloader, input_size, action_nextday = False, eps=0.5,  retrain = True, epoches=20) -> pd.Series:

    if os.path.exists(MODEL_PATH) and retrain == False:
        model = torch.load(MODEL_PATH)
    else:
        model = train_mlp(train_dataloader, input_size, hidden_size=8, output_classes=1, epoches=epoches)
        
    pred_reward = []
    pred_label = []
    for i, item in enumerate(test_dataloader):
        feature, labels = item
        outputs = model(feature)
        pred_reward.append(outputs.detach().numpy().flatten())   
        pred_label.append(labels.detach().numpy().flatten())   
    
    pred_reward_sequence = np.concatenate(pred_reward)
    pred_label_sequence = np.concatenate(pred_label)
    action_sequence = pd.Series(pred_reward_sequence).apply(lambda x: 1 if x < eps else -1)
    
    
    if action_nextday: # 第二天才action
        action_sequence = action_sequence.shift(1).fillna(0).astype("int")
    
    action_sequence = np.array(action_sequence)
    test_df = test_df.iloc[LookBack-1:,:]

    test_df['reward_'] = pred_reward_sequence # predict reward
    test_df['action'] = action_sequence # predict action
    
    
    return test_df
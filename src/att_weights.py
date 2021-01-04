import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from data import data_preprocess, data_trans

class STA_LSTM(nn.Module):
    
    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 use_gpu=False):

        super(STA_LSTM,self).__init__()

        # 参数导入部分

        self.in_dim = in_dim
        self.sequence_length = sequence_length
        
        self.lstm_in_dim = lstm_in_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.out_dim = out_dim
        self.use_gpu = use_gpu

        # 网络结构部分

        # batch_norm layer
        self.batch_norm = nn.BatchNorm1d(in_dim)
        
        # input layer
        self.layer_in = nn.Linear(in_dim, in_dim,bias=False)

        # spatial atteention module
        self.S_A = nn.Linear(lstm_in_dim, lstm_in_dim)
        
        # lstmcell 
        self.lstmcell = nn.LSTMCell(lstm_in_dim, lstm_hidden_dim)
        
        # temporal atteention module, 产生sequence_length个时间权重, 维度1 ×（lstm_hidden_dim + lstm_in_dim）-> 1 × sequence_length
        self.T_A = nn.Linear(sequence_length*lstm_hidden_dim, sequence_length)
        
        # # output layer, 维度 1 × lstm_hiddendim -> 1 × 1
        self.layer_out = nn.Linear(lstm_hidden_dim, out_dim,bias=False)
         
        # activate functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward (self,input):

        # 批归一化处理输入
        out = self.batch_norm(input)
        # print('batch_norm',out.size())

        # 经过输入层处理
        out = self.layer_in(out)
        # print('layer_in',out.size())
         
        # 初始化隐藏状态与记忆单元状态
        h_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim) # batch, hidden_size
        c_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim) # batch, hidden_size
      
        # 创建一个列表，存储ht
        h_list = []
        A = []
        for i in range(self.sequence_length):
            
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]
           
            alpha_t =  self.sigmoid(self.S_A(x_t))

            alpha_t = self.softmax(alpha_t)

            a = np.array(alpha_t.data.numpy()).reshape((len(alpha_t.data.numpy()[0])))
            A.append(a)
            h_t,c_t = self.lstmcell(x_t*alpha_t,(h_t_1,c_t_1)) 
            
            h_list.append(h_t)

            h_t_1,c_t_1 = h_t,c_t

        np.savetxt('./models/A.csv',A, delimiter=',')

        total_ht = h_list[0]
        for i in range(1,len(h_list)):
            total_ht = torch.cat((total_ht,h_list[i]),1)    
        
        beta_t =  self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)
        B = np.array(beta_t.data.numpy()).reshape((len(beta_t.data.numpy()[0])))
        np.savetxt('./models/B.csv',B, delimiter=',')
        out = torch.zeros(out.size(0), self.lstm_hidden_dim)
        # print(h_list[i].size(),beta_t[:,1].size())

        for i in range(len(h_list)):
                      
            out = out + h_list[i]*beta_t[:,i].reshape(out.size(0),1)

        out = self.layer_out(out)
        
        return out

'''****************************initialization*******************************''' 
IN_DIM =  96       # 因变量 TX144，CH96，HH120
SEQUENCE_LENGTH = 12   # 时间序列长度，即为回溯期

LSTM_IN_DIM = int(IN_DIM/SEQUENCE_LENGTH)     # LSTM的input大小,等于总的变量长度/时间序列长度
LSTM_HIDDEN_DIM = 300  # LSTM隐状态的大小

OUT_DIM = 1            # 输出大小

LEARNING_RATE = 0.1 # learning rate
WEIGHT_DECAY = 1e-6    # L2惩罚项

BATCH_SIZE = 200        # batch size

EPOCHES = 20     # epoch大小

TRAIN_PER = 0.80 # 训练集占比
VALI_PER = 0.0 # 验证集占比

# 判断是否采用GPU加速
# USE_GPU = torch.cuda.is_available()
USE_GPU = False

net = STA_LSTM(IN_DIM,SEQUENCE_LENGTH,LSTM_IN_DIM,LSTM_HIDDEN_DIM,OUT_DIM,USE_GPU)

trained_net = torch.load('./model/sta_lstm_t+1.pth')
net.load_state_dict(trained_net.state_dict())
net.eval()

sample_input = [[19,13,0.7,0.1,0,0.1,0,129,0,0,0.7,0.1,0,0.1,0,126,0,0,0.7,0.1,0,0.1,0,123,8,8,10,11.3,2,1.23,1,131,8,8,10,11.3,2,1.23,1,131,0,0,1.15,0.8,1,1.23,5,172.5,0,0,1.15,0.8,1,1.23,5,214,0,1,0.9,0.4,0,0.1,0,469,1,1,0.9,0.4,0,0.1,0,557.79,1,1,0.9,0.4,0,0.1,0,547.25,0,0,0.9,0.4,0,0.1,0,488.5,0,0,0.9,0.4,0,0.1,0,429.75]]

sample_input = torch.tensor(sample_input)
out = net(sample_input)
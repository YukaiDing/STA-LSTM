import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


class SA_LSTM(nn.Module):

    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 use_gpu=False):

        super(SA_LSTM,self).__init__()

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
      
        for i in range(self.sequence_length):
            
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]
           
            alpha_t =  self.sigmoid(self.S_A(x_t))

            alpha_t = self.softmax(alpha_t)

            h_t,c_t = self.lstmcell(x_t*alpha_t,(h_t_1,c_t_1)) 

            h_t_1,c_t_1 = h_t,c_t
        
        out = self.layer_out(h_t)
        
        return out

class TA_LSTM(nn.Module):

    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 use_gpu=False):

        super(TA_LSTM,self).__init__()

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

        for i in range(self.sequence_length):
            
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]
           
            h_t,c_t = self.lstmcell(x_t,(h_t_1,c_t_1)) 
            
            h_list.append(h_t)

            h_t_1,c_t_1 = h_t,c_t
        
        total_ht = h_list[0]
        for i in range(1,len(h_list)):
            total_ht = torch.cat((total_ht,h_list[i]),1)    
        
        beta_t =  self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)
        
        out = torch.zeros(out.size(0), self.lstm_hidden_dim)
        # print(h_list[i].size(),beta_t[:,1].size())

        for i in range(len(h_list)):
                      
            out = out + h_list[i]*beta_t[:,i].reshape(out.size(0),1)

        out = self.layer_out(out)
        
        return out
        
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

        for i in range(self.sequence_length):
            
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]
           
            alpha_t =  self.sigmoid(self.S_A(x_t))

            alpha_t = self.softmax(alpha_t)
            # print(alpha_t)

            h_t,c_t = self.lstmcell(x_t*alpha_t,(h_t_1,c_t_1)) 
            
            h_list.append(h_t)

            h_t_1,c_t_1 = h_t,c_t
        
        total_ht = h_list[0]
        for i in range(1,len(h_list)):
            total_ht = torch.cat((total_ht,h_list[i]),1)    
        
        beta_t =  self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)
        # print(beta_t)
        out = torch.zeros(out.size(0), self.lstm_hidden_dim)
        # print(h_list[i].size(),beta_t[:,1].size())

        for i in range(len(h_list)):
                      
            out = out + h_list[i]*beta_t[:,i].reshape(out.size(0),1)

        out = self.layer_out(out)
        
        return out

class LSTM(nn.Module):
    
    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 use_gpu=False):

        super(LSTM,self).__init__()

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
        self.layer_in = nn.Linear(in_dim, in_dim)
        
        # lstmcell 
        self.lstmcell = nn.LSTMCell(lstm_in_dim, lstm_hidden_dim)
         
        # # output layer, 维度 1 × lstm_hiddendim -> 1 × 1
        self.layer_out = nn.Linear(lstm_hidden_dim, out_dim,bias=False)
        
        # activate functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward (self,input):

        # 批归一化处理输入
        out = self.batch_norm(input)
        # print('batch_norm',out.size())

        # 经过输入层处理
        out = self.layer_in(out)
        
        # print('layer_in',out.size())

        # 初始化隐藏状态与记忆单元状态
        h_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim) # batch, hideden_size
        c_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim) # batch, hideden_size
        
        for i in range(self.sequence_length):
            
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]

            h_t,c_t = self.lstmcell(x_t,(h_t_1,c_t_1)) 
            h_t_1,c_t_1 = h_t,c_t
        
        out = self.layer_out(h_t)
                
        return out

class FCN(nn.Module):
    
    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 use_gpu=False):

        super(FCN, self).__init__()

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

        # fcn
        self.fcn = nn.Linear(in_dim, lstm_hidden_dim)
         
        # # output layer, 维度 1 × lstm_hiddendim -> 1 × 1
        self.layer_out = nn.Linear(lstm_hidden_dim, out_dim,bias=False)

        # activate functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward (self,input):

        # 批归一化处理输入
        out = self.batch_norm(input)
        # print('batch_norm',out.size())

        # 经过输入层处理
        out = self.layer_in(out)
        # print('layer_in',out.size())

        out = self.sigmoid(out)

        out = self.fcn(out)

        out = self.sigmoid(out)

        out = self.layer_out(out)
      
        return out

        
def main():

    in_dim = 72
    sequence_length = 6
    lstm_in_dim = 12
    lstm_hidden_dim = 64
    out_dim = 1
    batch_size = 200
    input = torch.randn(batch_size, 72)

    batch_norm = nn.BatchNorm1d(in_dim)
    in_linear = nn.Linear(in_dim, in_dim)
    
    out = batch_norm(input)
    print(out.size())

    out = in_linear(out)
    print(out.size())
    
    h = torch.zeros(batch_size, lstm_hidden_dim)
    c = torch.zeros(batch_size, lstm_hidden_dim) # batch,hideden_size

    lstmcell = nn.LSTMCell(lstm_in_dim, lstm_hidden_dim)

    for i in range(sequence_length):

        temp_out = out[:,i*lstm_in_dim:(i+1)*(lstm_in_dim)]

        print(temp_out.size())

        h,c = lstmcell(temp_out,(h,c))

        print(h.size())
        print(h)


if __name__ == '__main__':
    main()



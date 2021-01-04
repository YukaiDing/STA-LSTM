import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# 实现数据读取，数据集划分以及数据输出

class data_preprocess(object):

    def __init__(self, file_path = None, train_per = 0.80, vali_per = 0.0, in_dim = None):

        self.file_path = file_path
        self.train_per = train_per
        self.vali_per = vali_per
        self.in_dim = in_dim
        
    # 读取csv文件的值
    def load_data(self):
        
        raw_data = pd.read_csv(self.file_path).values
        
        # print('raw_data.shape = ',raw_data.shape)

        return raw_data
    
    # split_data方法将数据分割为训练、验证与测试数据，每类数据包含因变量和真值
    def  split_data(self, raw_data = None, _type = 'linear' ):
        
        length = len(raw_data)
        # print('数据集总量:{}'.format(length))
        
        train_set_size = int(length*self.train_per)
        vali_set_size = int(length*self.vali_per)

        test_set_size = int(length-train_set_size-vali_set_size)

        # print('训练集数量:{}'.format(train_set_size),'\n验证集数量:{}'.format(vali_set_size),'\n测试集数量:{}'.format(test_set_size))
        # print('训练集占比:{}%'.format(self.train_per*100),'\n验证集占比:{}%'.format(self.vali_per*100),'\n测试集占比:{}%'.format((1-self.train_per-self.vali_per)*100))  

        
        if _type == 'linear':

            train_data = raw_data[:train_set_size, :self.in_dim]
            vali_data = raw_data[train_set_size: train_set_size+vali_set_size, :self.in_dim]
            test_data = raw_data[train_set_size + vali_set_size:, :self.in_dim]

            train_groundtruth = raw_data[:train_set_size, self.in_dim:]
            vali_groundtruth = raw_data[train_set_size: train_set_size+vali_set_size, self.in_dim:]
            test_groundtruth = raw_data[train_set_size + vali_set_size:, self.in_dim:]
        else:

            pass

        return (train_data,train_groundtruth),(vali_data,vali_groundtruth),(test_data,test_groundtruth)

    
    # # 将获得所有数据划分为训练集和测试集，并输出到外部csv文件
    # def save_data(self, train_file_path = './train_set.csv', test_file_path = './test_set.csv'):
                       
    #     train_set = pd.DataFrame([[0,0],[0,0]],index=['row1','row2'],columns=['column1','column2'])
    #     test_set = pd.DataFrame([[0,0],[0,0]],index=['row1','row2'],columns=['column1','column2'])
         
    #     train_set.to_csv(train_file_path,encoding = 'utf-8')
    #     test_set.to_csv(test_file_path,encoding = 'utf-8')

# 实现数据类型的转换    
class data_trans(Dataset):
        
    def __init__(self, data, groundtruth, transform=None):

        self.data = self._get_data(data)
        self.groundtruth = self._get_data(groundtruth)
        self.transform = transform

    def _get_data(self,data):
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
               
        inputs = self.data[index,:]
        groundtruths = self.groundtruth[index,:]
        
        #将数据从numpy.array转换为tensor类型       
        if self.transform:
                        
            inputs = torch.from_numpy(inputs).float()
            groundtruths = torch.from_numpy(groundtruths).float()
                       
        return {'inputs':inputs,'groundtruths':groundtruths}

def main():
    # pass
    dp = data_preprocess(file_path = './data/dataset/sample_t+1.csv',train_per = 0.8, vali_per = 0.0, in_dim = 96)
    raw_data = dp.load_data()
    (train_data,train_groundtruth),(vali_data,vali_groundtruth),(test_data,test_groundtruth) = dp.split_data(raw_data = raw_data, _type = 'linear')
    print("done!")

if __name__ == '__main__':
    main()    
    
        
        



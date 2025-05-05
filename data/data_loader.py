

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
class Dataset_ETTh(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path).values
        raw_data = data[:, 1:data.shape[1]]
        df = pd.DataFrame(raw_data)

        self.data = df.dropna(axis=0, how='any').values


        self.scaler = StandardScaler()
        if type == '1': 


            train_data = self.data[0:8449]

            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data)



        if self.flag == 'train':
            begin = 0
            end = 8449
            self.trainData = self.data[begin:end]
            self.train_nextData = self.data[begin:end]
        if self.flag == 'val':
            begin = 8449
            end = 11234
            self.valData = self.data[begin:end]
            self.val_nextData = self.data[begin:end]
        if self.flag == 'test':
            begin = 11234
            end = 14019
            self.testData = self.data[begin:end]
            self.test_nextData = self.data[begin:end]

    def __getitem__(self, index):
        # data timestamp
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        # return the time data , next time data and time
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len
class Dataset_CSI(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path)
        data['date'] = pd.to_datetime(data['date'])
        raw_data = data.iloc[:, 1:].values#.

        df = pd.DataFrame(raw_data)


        self.data = df.dropna(axis=0, how='any').values

        self.scaler = StandardScaler()
        if type == '1':


            train_data = self.data[0:int(len(self.data) * self.train_ratio)]

            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data)

        if self.flag == 'train':
            begin = 0
            end = int(len(self.data)*self.train_ratio)
            self.trainData = self.data[begin:end]
            print(self.trainData.shape)
            self.train_nextData = self.data[begin:end] 
        if self.flag == 'val':
            begin = int(len(self.data)*self.train_ratio)
            end = int(len(self.data)*(self.train_ratio+self.val_ratio))
            self.valData = self.data[begin:end]
            self.val_nextData = self.data[begin:end] 
        if self.flag == 'test':
            begin = int(len(self.data)*(self.train_ratio+self.val_ratio))
            end = len(self.data)
            self.testData = self.data[begin:end]
            self.test_nextData = self.data[begin:end] 

    def __getitem__(self, index):
        # data timestamp
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        # return the time data , next time data and time
        return data, next_data
    def __len__(self):
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ECG(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path)

        self.scaler = StandardScaler()
        if type == '1': 
            train_data = data[0:int(len(data) * self.train_ratio)]

            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        data = np.array(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data)*self.train_ratio)
            end = int(len(data)*(self.val_ratio+self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data)*(self.val_ratio+self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

class Dataset_CSI_old(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):  
        assert flag in ['train', 'test', 'val']  
        self.path = root_path  
        self.flag = flag  
        self.seq_len = seq_len  
        self.pre_len = pre_len  
        self.train_ratio = train_ratio  
        self.val_ratio = val_ratio  
        data = pd.read_csv(root_path)  
        data['date'] = pd.to_datetime(data['date'])
        raw_data = data.iloc[:, 1:].values   
        df = pd.DataFrame(raw_data)  

        self.data = df.dropna(axis=0, how='any').values  
        self.scaler = StandardScaler()
        if type == '1':  
            mms = MinMaxScaler(feature_range=(0, 1))
            self.data = mms.fit_transform(self.data)

        if self.flag == 'train':  
            begin = 0  
            end = int(len(self.data) * self.train_ratio)  
            self.trainData = self.data[begin:end]  
            print(self.trainData.shape)
            self.train_nextData = self.data[begin:end]  
        if self.flag == 'val':  
            begin = int(len(self.data) * self.train_ratio)  
            end = int(len(self.data) * (self.train_ratio + self.val_ratio))  
            self.valData = self.data[begin:end]  
            self.val_nextData = self.data[begin:end]  
        if self.flag == 'test':  
            begin = int(len(self.data) * (self.train_ratio + self.val_ratio))  
            end = len(self.data)  
            self.testData = self.data[begin:end]  
            self.test_nextData = self.data[begin:end]  

    def __getitem__(self, index):  
        # data timestamp
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        # return the time data , next time data and time
        return data, next_data

    def __len__(self): 
        if self.flag == 'train':
            return len(self.trainData) - self.seq_len - self.pre_len
        elif self.flag == 'val':
            return len(self.valData) - self.seq_len - self.pre_len
        else:
            return len(self.testData) - self.seq_len - self.pre_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
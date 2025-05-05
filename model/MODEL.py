import pandas as pd
import torch
from torch import nn
from model.SGSC import SGSC
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F



class DSTSGNN(nn.Module):
    def __init__(self, stride, pre_length, embed_size, feature_size, seq_length, hidden_size, patch_len, d_model):
        super(DSTSGNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.stride = stride
        self.patch_len = patch_len
        self.d_model = d_model
        
        self.moving_avg = 2
        self.decompsition = series_decomp(self.moving_avg)

        self.model1s = SGSC(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size,
                               self.patch_len, self.d_model)
        self.model1t = SGSC(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size,
                               self.patch_len, self.d_model)

        self.fc1 = nn.Linear(self.embed_size*12, 512).double()
        self.fc2 = nn.Linear(512, self.hidden_size).double()
        self.fc3 = nn.Linear(self.hidden_size, self.pre_length).double()

        self.cuda(0)

    def forward(self, x):
        x = x.float()

        z1 = x
        z1 = z1.permute(0, 2, 1)
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=24)


        zz1 = z1.reshape(z1.shape[0], z1.shape[1] * z1.shape[2], z1.shape[3])
        season1, trend1 = self.decompsition(zz1)
        m1s = season1.permute(0, 2, 1)
        m1t = trend1.permute(0, 2, 1)
        F1s = self.model1s(m1s)
        F1t = self.model1t(m1t)
        F1 = F1s + F1t
        F1 = F1.reshape(z1.shape[0], z1.shape[1], self.embed_size*12)
        F1 = F1.double()
        Y1 = self.fc1(F1)
        Y1 = self.fc2(Y1.detach().clone()).double()
        Y1 = self.fc3(Y1.detach().clone())

        return Y1


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):

        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2 + 1)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)

        x = self.avg(x)  
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

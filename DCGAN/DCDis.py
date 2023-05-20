"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/26 15:28
"""

import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

class Discriminator(torch.nn.Module):
    def __init__(self,features = 64):
        super(Discriminator, self).__init__()
        self.in_features = features
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3,out_channels = features,kernel_size = (4,4),
                            stride=(2,2),padding = (1,1),bias = False),
            torch.nn.BatchNorm2d(num_features= features),
            torch.nn.LeakyReLU(negative_slope = 0.2,inplace = True),#inplace，其作用是：该nn.Relu() 函数计算得到的输出是否更新传入的输出。

            torch.nn.Conv2d(in_channels=features , out_channels=features * 2, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.LeakyReLU(negative_slope = 0.2,inplace = True),

            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.LeakyReLU(negative_slope = 0.2,inplace = True),

            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 8, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.LeakyReLU(negative_slope = 0.2,inplace = True),

            torch.nn.Conv2d(in_channels=features * 8, out_channels=1, kernel_size=(4, 4),
                            stride=(1,1), padding=(0,0), bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self,input):
        x = self.conv(input)
        return x

if __name__ == '__main__':
    dis = Discriminator()
    print(dis)
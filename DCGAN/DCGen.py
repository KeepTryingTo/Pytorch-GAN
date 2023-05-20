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

class Generator(torch.nn.Module):
    def __init__(self,randomSize = 100,features = 64,channels = 3):
        super(Generator, self).__init__()
        self.randomNoise = randomSize#输入到网络随机噪声向量大小
        self.in_features = features#输入的特征数
        self.channels = channels#最后网络输出的通道数
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels = randomSize,out_channels = features * 8,kernel_size = (4,4),
                            stride=(1,1),padding = (0,0),bias = False),
            torch.nn.BatchNorm2d(num_features= features * 8),
            torch.nn.ReLU(inplace = True),#inplace，其作用是：该nn.Relu() 函数计算得到的输出是否更新传入的输出。

            torch.nn.ConvTranspose2d(in_channels=features * 8, out_channels=features * 4, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(in_channels=features * 4, out_channels=features * 2, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(in_channels=features * 2, out_channels=features, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(in_channels=features, out_channels=channels, kernel_size=(4, 4),
                            stride=(2, 2), padding=(1,1), bias=False),
            torch.nn.Tanh()
        )
    def forward(self,randomNoise):
        img = self.conv(randomNoise)#(channels,64,64)
        return img

if __name__ == '__main__':
    gen = Generator()
    print(gen)
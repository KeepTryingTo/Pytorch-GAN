"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/24 14:21
"""
import torch
import numpy as np
#对于生成器，输入的为正态分布随机数
#输出为: [1,28,28]图片

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=100,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512,out_features=784),
            torch.nn.Tanh()#对于生成器使用tanh激活函数更好
        )
    def forward(self,input):
        x = self.fc(input)
        img = x.view(-1,28,28)
        return img
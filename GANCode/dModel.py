"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/24 14:21
"""
import torch
import numpy as np

#判别器的输入为一张图片
#输出为二分类的概率值
#判别器对log(1 - D(G(z)))的判别作为生成器的损失值

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=784,out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256,out_features=1),
            torch.nn.Sigmoid()
        )
    def forward(self,input):
        x = input.view(-1,784)
        x = self.fc(x)
        return x
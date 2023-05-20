"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/11 16:25
"""

import torch
from torchinfo import summary

class Discrimiantor(torch.nn.Module):
    def __init__(self,in_features = 128):
        super(Discrimiantor, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features,out_features=256),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256,out_features=512),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features = 512,out_features=256),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features = 256,out_features=128),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features = 128,out_features=1),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        out = self.dense(x)
        return out

if __name__ == '__main__':
    model = Discrimiantor(in_features=128)
    #注意这个地方的input_size不能为input_size=(128)
    summary(model,input_size=(128,))
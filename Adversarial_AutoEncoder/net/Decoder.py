"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/11 16:36
"""

import torch
from torchinfo import summary

class Decoder(torch.nn.Module):
    def __init__(self,in_features = 784,out_features = 128):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=out_features, out_features=256),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=512, out_features=in_features)
        )
    def forward(self,x):
        out = self.decoder(x)
        out = out.view(-1,28,28)
        return out

if __name__ == '__main__':
    model = Decoder(in_features=784,out_features=128)
    summary(model,input_size=(128,))
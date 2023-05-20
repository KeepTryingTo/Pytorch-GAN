"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/6 16:22
"""
import numpy as np
import torch
from torchinfo import summary

class AE(torch.nn.Module):
    def __init__(self,in_feautres = 784,out_features = 128):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_feautres,out_features=512),
            torch.nn.Dropout(p = 0.5),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256,out_features=out_features),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=out_features, out_features=256),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=512, out_features=in_feautres)
        )
    def forward(self,x):
        x = x.view(-1,784)
        e_x = self.encoder(x)
        d_x = self.decoder(e_x)
        img = d_x.view(-1,28,28)
        return img

if __name__ == '__main__':
    x= torch.randn(size = (1,28,28),device='cpu')
    encoder_Decoder = AE(in_feautres=784,out_features=128)
    print(encoder_Decoder(x).shape)
    summary(encoder_Decoder,input_size=(1,28,28))
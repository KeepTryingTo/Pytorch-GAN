"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/19 9:33
"""

import os
import torch

class Generator(torch.nn.Module):
    def __init__(self,in_features = 100,W = 28,H = 28):
        super(Generator, self).__init__()
        self.fc_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features,out_features=256),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU()
        )
        self.fc_layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=10,out_features=256),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU()
        )
        self.fc_layer_3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=512),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU()
        )
        self.fc_layer_4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=1024),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU()
        )
        self.fc_layer_final = torch.nn.Linear(in_features=1024,out_features=H * W)

    def forward(self,input,label):
        x = self.fc_layer_1(input)
        y = self.fc_layer_2(label)
        x = torch.cat([x,y],dim=1)
        x = self.fc_layer_3(x)
        x = self.fc_layer_4(x)
        out = torch.tanh(self.fc_layer_final(x))
        return out

class Discriminator(torch.nn.Module):
    def __init__(self,W = 28,H = 28,out_features = 1):
        super(Discriminator, self).__init__()
        self.fc_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=W * H,out_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.fc_layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=10,out_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.fc_layer_3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048,out_features=512),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.fc_layer_4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.fc_final_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self,input,label):
        x = self.fc_layer_1(input.view(input.size(0),-1))
        y = self.fc_layer_2(label)
        x = torch.cat([x,y],dim = 1)
        x = self.fc_layer_3(x)
        x = self.fc_layer_4(x)
        out = self.fc_final_layer(x)
        return out


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight,0.0,0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
        torch.nn.init.constant(m.bias,0.0)

if __name__ == '__main__':
    gen = Generator()
    disc = Discriminator()

    x = torch.randn(size = (2,100)).view(-1,100)
    y = torch.zeros(2,10).view(-1,10)

    gen_out = gen(x,y)
    print('gen_out.shape: {}'.format(gen_out.shape))

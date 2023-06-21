"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/19 9:33
"""

import os
import torch
import config
import numpy as np
from torchinfo import summary



class Generator(torch.nn.Module):
    def __init__(self,img_channels = 1,d = 128):
        super(Generator, self).__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=100,
                out_channels=d * 2,
                kernel_size=(4,4),
                stride=(1,1),
                padding=(0,0)
            ),
            torch.nn.BatchNorm2d(num_features=2 * d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=10,
                out_channels=d * 2,
                kernel_size=(4, 4),
                stride=(1,1),
                padding=(0,0)
            ),
            torch.nn.BatchNorm2d(num_features=2 * d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=4 * d,
                out_channels=d * 2,
                kernel_size=(4, 4),
                stride=(2,2),
                padding=(1,1)
            ),
            torch.nn.BatchNorm2d(num_features=2 * d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=d * 2,
                out_channels=d,
                kernel_size=(4, 4),
                stride=(2,2),
                padding=(1,1)
            ),
            torch.nn.BatchNorm2d(num_features=d),
            torch.nn.ReLU(inplace=True)
        )
        self.final_conv_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=d,
                out_channels=img_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            )
        )
    def forward(self,input,label):
        x = self.conv_layer_1(input)
        y = self.conv_layer_2(label)
        x = torch.cat([x,y], dim = 1)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        out = torch.tanh(self.final_conv_layer(x))
        return out

class Discriminator(torch.nn.Module):
    def __init__(self,img_channels = 1,out_channels = 1,d = 128):
        super(Discriminator, self).__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=img_channels,
                out_channels=d // 2,
                kernel_size=(4,4),
                stride=(2,2),
                padding=(1,1)
            ),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=10,
                out_channels=d // 2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=d,
                out_channels=d * 2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(num_features=d * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=d * 2,
                out_channels=d * 4,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(num_features=d * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.final_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=d * 4,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=(1,1),
                padding=(0,0)
            ),
            torch.nn.Sigmoid()
        )
    def forward(self,input,label):
        x = self.conv_layer_1(input)
        y = self.conv_layer_2(label)
        x = torch.cat([x,y], dim = 1)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        out = self.final_layer(x)
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

    x = torch.randn(size = (2,100,1,1)).view(-1,100,1,1)
    y = torch.zeros(2,10).view(-1,10,1,1)

    gen_out = gen(x,y).view(2,1,32,32)
    print('gen_out.shape: {}'.format(gen_out.shape))
    # summary(gen,input_size=([[1,100,1,1],[1,10,1,1]]))

    disc_out = disc(gen_out,y.expand(-1,-1,config.IMG_SIZE,config.IMG_SIZE))
    print('disc_out.shape: {}'.format(disc_out.shape))
    print('y.expand.shape: {}'.format(np.shape(y.expand(-1,-1,config.IMG_SIZE,config.IMG_SIZE))))

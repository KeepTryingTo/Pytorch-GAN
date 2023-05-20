"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/28 17:14
"""

import torch
import torchvision
from torchinfo import summary
from torchvision import transforms

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,down=True,use_act=True,**kwargs):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,padding_mode='reflect',**kwargs)
            if down
            else torch.nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,**kwargs),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True) if use_act else torch.nn.Identity()
        )
    def forward(self,x):
        return self.conv(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.block = torch.nn.Sequential(
            ConvBlock(channels,channels,kernel_size = 3,padding = 1),
            ConvBlock(channels,channels,use_act=False,kernel_size = 3,padding = 1),
        )
    def forward(self,x):
        return x + self.block(x)

class Generator(torch.nn.Module):
    def __init__(self,img_channels,num_features = 64,num_residual=9):
        super(Generator, self).__init__()
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_channels,out_channels=num_features,kernel_size=(7,7),stride=(1,1),padding=3,padding_mode='reflect'),
            torch.nn.ReLU(inplace=True)
        )
        self.down_blocks = torch.nn.ModuleList(
            [
                ConvBlock(in_channels=num_features,out_channels=num_features*2,kernel_size=(3,3),stride=(2,2),padding=1),
                ConvBlock(in_channels=num_features*2, out_channels=num_features * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)
            ]
        )
        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residual)]
        )
        self.up_blocks = torch.nn.ModuleList(
            [
                ConvBlock(in_channels=num_features*4,out_channels=num_features*2,down=False,kernel_size=3,stride = 2,padding=1,output_padding=1),
                ConvBlock(in_channels=num_features * 2, out_channels=num_features * 1, down=False, kernel_size=3, stride=2, padding=1,output_padding=1)
            ]
        )
        self.last = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_features, out_channels=img_channels, kernel_size=(7, 7), stride=(1, 1),
                            padding=3, padding_mode='reflect'),
            torch.nn.Tanh()
        )
    def forward(self,x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        out = self.last(x)
        return out

if __name__ == '__main__':
    img_channels = 3
    img_size = 256
    x = torch.randn(size = (2,img_channels,img_size,img_size))
    model = Generator(img_channels,9)
    summary(model,input_size=(2,img_channels,img_size,img_size))
    print(model(x).shape)
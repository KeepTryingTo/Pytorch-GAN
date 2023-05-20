"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/28 17:14
"""

import torch
import torchvision
from torchvision import transforms

class Block(torch.nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4),
                            stride=stride,padding=1,bias=True,padding_mode='reflect'),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
    def forward(self,x):
        out = self.conv(x)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3,features=(64,128,256,512)):
        super(Discriminator, self).__init__()
        self.features = features
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=(4, 4),
                            stride=(2,2), padding=1, bias=True, padding_mode='reflect'),
            torch.nn.BatchNorm2d(num_features=features[0]),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        layers = []
        in_channels=features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels,feature,stride=1 if feature == features[-1] else 2)
            )
            in_channels=feature
        layers.append(torch.nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=(4,4),
                                      stride=(1,1),padding=1,padding_mode='reflect'))
        #将值归一化到[0-1]
        layers.append(torch.nn.Sigmoid())
        #对layers进行解序列
        self.model = torch.nn.Sequential(
            *layers
        )
    def forward(self,x):
        x = self.initial(x)
        out= self.model(x)
        return out

if __name__ == '__main__':
    x = torch.randn(size = (5,3,256,256),device='cpu')
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 12:08
"""

import torch
import torchvision
from torchinfo import summary
from torchvision import transforms

class Discriminator(torch.nn.Module):
    def __init__(self,channels_img,features_d):
        super(Discriminator, self).__init__()
        self.disc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channels_img,out_channels=features_d,kernel_size=(4,4),stride=(2,2),padding=(1,1)
            ),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True),
            self._block(in_channels=features_d,out_channels=features_d * 2,kernel_size=(4,4),stride=(2,2),
                        padding=(1,1)),
            self._block(in_channels=features_d * 2, out_channels=features_d * 4, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            self._block(in_channels=features_d * 4, out_channels=features_d * 8, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            torch.nn.Conv2d(in_channels=features_d*8,out_channels=1,kernel_size=(4,4),stride=(2,2),padding=0)
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            #affine=True:一个布尔值，当设置为True时，该模块具有可学习的仿射参数，以与批量规范化相同的方式初始化。默认值：False。
            torch.nn.InstanceNorm2d(num_features=out_channels,affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        return self.conv
    def forward(self,input):
        x = self.disc(input)
        return x

if __name__ == '__main__':
    in_channles = 3
    H,W = 64,64
    x = torch.randn(size = (1,in_channles,H,W))
    disc = Discriminator(in_channles,features_d=8)
    summary(disc, input_size=(1, in_channles, H,W))
    print(disc(x).shape)
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

class Generator(torch.nn.Module):
    def __init__(self,channels_noise,channels_img,features_g):
        super(Generator, self).__init__()
        self.gen = torch.nn.Sequential(
            #imgsize: 4 x 4
            self._block(in_channels = channels_noise,out_channels = features_g * 16,kernel_size = (4,4),
                        stride=(1,1),padding=0),
            # imgsize: 8 x 8
            self._block(in_channels=features_g * 16, out_channels=features_g * 8, kernel_size=(4, 4), stride=(2,2),
                        padding=1),
            # imgsize: 16 x 16
            self._block(in_channels=features_g * 8, out_channels=features_g * 4, kernel_size=(4, 4), stride=(2,2),
                        padding=1),
            # imgsize: 32 x 32
            self._block(in_channels=features_g * 4, out_channels=features_g * 2, kernel_size=(4, 4), stride=(2,2),
                        padding=1),
            # imgsize: N x 3 x 64 x 64
            torch.nn.ConvTranspose2d(
                in_channels=features_g * 2, out_channels=channels_img, kernel_size=(4,4), stride=(2,2),
                padding=(1,1)
            ),
            torch.nn.Tanh()
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU()
        )
        return self.conv

    def forward(self,input):
        x = self.gen(input)
        return x

if __name__ == '__main__':
    noise_dim = 100
    in_channels = 3
    feature_g = 8
    gen = Generator(noise_dim,in_channels,feature_g)
    z = torch.randn(size = (1,noise_dim,1,1))
    summary(gen,input_size=(1,noise_dim,1,1))
    print(gen(z).shape)
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/15 20:52
"""

import torch
from torchinfo import summary

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,
                 discriminator = False,use_act = True,
                 use_bn = True,**kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.cnn = torch.nn.Conv2d(in_channels,out_channels,**kwargs,bias=not use_bn)
        #Identity()表示输入是什么输出就是什么
        #要加深网络，有些层是不改变输入数据的维度的，
        #在增减网络的过程中我们就可以用identity占个位置，这样网络整体层数永远不变，
        self.bn = torch.nn.BatchNorm2d(out_channels) if use_bn else torch.nn.Identity()
        #对于generator使用PReLU
        #对于discriminator使用LeakReLU
        self.act = (
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
            if discriminator else torch.nn.PReLU(num_parameters=out_channels)
        )
    def forward(self,x):
        out = self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
        return out

#采用PixelShuffle进行上采样
class UpsampleBlock(torch.nn.Module):
    def __init__(self,in_channels,scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,in_channels*scale_factor**2,kernel_size = (3,3),stride = (1,1),padding=(1,1))
        #(in_channels * 4,H,W) => (in_channels,H*2,W*2)
        self.ps = torch.nn.PixelShuffle(scale_factor)
        self.act = torch.nn.PReLU(num_parameters=in_channels)
    def forward(self,x):
        out = self.act(self.ps(self.conv(x)))
        return  out


class ResidualBlock(torch.nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1)
        )
        #block2不使用激活函数
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
            use_act = False
        )
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x

class Generator(torch.nn.Module):
    def __init__(self,in_channels = 3, num_channels = 64,num_blocks = 16):
        super(Generator, self).__init__()
        self.initial = ConvBlock(in_channels,num_channels,kernel_size = (9,9),stride = (1,1),padding = (4,4),use_bn=False)
        self.residuals = torch.nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.convblock = ConvBlock(num_channels,num_channels,kernel_size = (3,3),stride = (1,1),padding = (1,1),use_act=False)
        self.upsample = torch.nn.Sequential(
            UpsampleBlock(num_channels,2),
            UpsampleBlock(num_channels,2)
        )
        self.final = torch.nn.Conv2d(num_channels,in_channels,kernel_size=(9,9),stride=(1,1),padding = (4,4))

    def forward(self,x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x= self.convblock(x) + initial
        x = self.upsample(x)
        return torch.tanh(self.final(x))

class Discriminator(torch.nn.Module):
    def __init__(self,in_channels = 3,features = [64,64,128,128,256,256,512,512]):
        super(Discriminator, self).__init__()
        blocks = []
        for idx,feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size = (3,3),
                    stride = (1 + idx % 2,1 + idx % 2),
                    padding = (1,1),
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True
                )
            )
            in_channels = feature
        self.blocks = torch.nn.Sequential(*blocks)
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(6,6)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=512 * 6 * 6,out_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True),
            torch.nn.Linear(in_features=1024,out_features=1)
        )
    def forward(self,x):
        out = self.blocks(x)
        return self.classifier(out)

if __name__ == '__main__':
    #96 x 96 => 24 x 24
    low_resolution = 24
    # x = torch.randn(size = (5,3,low_resolution,low_resolution))
    x = torch.randn(size=(5, 3, low_resolution, low_resolution ))
    gen = Generator()
    gen_out = gen(x)
    disc = Discriminator()
    disc_out = disc(x)

    #display the generator's and discriminator's network
    # summary(gen, input_size=(1,3,low_resolution,low_resolution))
    # summary(disc,input_size=(1,3,low_resolution,low_resolution))

    #print the shape
    print('gen_out.shape: {}'.format(gen_out.shape))
    print('disc_out.shape: {}'.format(disc_out.shape))
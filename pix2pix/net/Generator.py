"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/4 13:28
"""

import torch
from torchinfo import summary

class Block(torch.nn.Module):
    def __init__(self,in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4),
                            stride=(2,2),padding=(1,1),bias=False,padding_mode='reflect')
            if down
            else torch.nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,
                                          kernel_size=(4,4),stride=(2,2),padding=(1,1),bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU() if act == "relu" else torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.conv(x)
        x = self.dropout(x) if self.use_dropout else x
        return x

class Generator(torch.nn.Module):
    def __init__(self,in_channles=3,features=64):
        super(Generator, self).__init__()
        self.initial_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channles,out_channels=features,kernel_size=(4,4),
                            stride=(2,2),padding=(1,1),padding_mode='reflect'),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.down1 = Block(in_channels=features,out_channels=features*2,down=True,act='leaky',use_dropout=False)
        self.down2 = Block(in_channels=features*2, out_channels=features * 4, down=True, act='leaky', use_dropout=False)
        self.down3 = Block(in_channels=features*4, out_channels=features * 8, down=True, act='leaky', use_dropout=False)
        self.down4 = Block(in_channels=features*8, out_channels=features * 8, down=True, act='leaky', use_dropout=False)
        self.down5 = Block(in_channels=features*8, out_channels=features * 8, down=True, act='leaky', use_dropout=False)
        self.down6 = Block(in_channels=features*8, out_channels=features * 8, down=True, act='leaky', use_dropout=False)

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8,out_channels=features*8,kernel_size=(4,4),
                            stride=(2,2),padding=(1,1),padding_mode='reflect'),
            torch.nn.ReLU()
        )

        self.up1 = Block(in_channels=features*8,out_channels=features*8,down=False,act="relu",use_dropout=True)
        self.up2 = Block(in_channels=features * 8*2, out_channels=features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(in_channels=features * 8*2, out_channels=features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(in_channels=features * 8*2, out_channels=features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(in_channels=features * 8*2, out_channels=features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(in_channels=features * 4*2, out_channels=features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(in_channels=features * 2*2, out_channels=features , down=False, act="relu", use_dropout=False)

        self.final_up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=features*2,out_channels=in_channles,kernel_size=(4,4),
                                     stride=(2,2),padding=(1,1)),
            torch.nn.Tanh()
        )
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1,d7],dim=1))
        u3 = self.up3(torch.cat([u2,d6],dim=1))
        u4 = self.up4(torch.cat([u3,d5],dim=1))
        u5 = self.up5(torch.cat([u4,d4],dim=1))
        u6 = self.up6(torch.cat([u5,d3],dim=1))
        u7 = self.up7(torch.cat([u6,d2],dim=1))

        final_up = self.final_up(torch.cat([u7,d1],dim=1))
        return final_up


if __name__ == '__main__':
    gen = Generator(in_channles=3,features=64)
    x = torch.randn(size = (1,3,256,256))
    print(gen(x).shape)
    summary(gen,input_size=(1,3,256,256))




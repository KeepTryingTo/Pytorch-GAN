"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/15 13:25
"""

import torch
from math import log2
from torchinfo import summary
import torch.nn.functional as F

#B的每个通道的缩放因子
factors = [1,1,1,1,1 / 2,1 / 4,1 / 8,1 / 16,1 / 32]

#Pixel norm沿着channel维度做归一化（axis=1），这样归一化的一个好处在于，
# feature map的每个位置都具有单位长度。
class PixelNorm(torch.nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
    def forward(self,x):
        #keepdim:输出张量是否保留了dim
        out = x / torch.sqrt(torch.mean(x**2,dim = 1,keepdim=True) + self.epsilon)
        return out

#紧跟PixelNorm之后的3 x 3卷积
class WSConv2d(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1,
                 padding = 1,gain = 2):
        super(WSConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.scale = (gain / (in_channels * (kernel_size**2)))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        #initialize conv layer
        #torch.init.normal_：给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。
        #torch.init.normal_(tensor,mean=,std=) ,mean:均值，std:正态分布的标准差
        torch.nn.init.normal_(self.conv.weight)
        #将其偏置设置为0
        torch.nn.init.zeros_(self.bias)

    def forward(self,x):
        out = self.conv(x * self.scale) + self.bias.view(1,self.bias.shape[0],1,1)
        # print(out.shape)
        return out

class WSLinear(torch.nn.Module):
    def __init__(self,in_features,out_features,gain = 2):
        super(WSLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features,out_features)
        self.scale = (gain / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        #initialize linear layer
        torch.nn.init.normal_(self.linear.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self,x):
        out = self.linear(x * self.scale) + self.bias
        return out


#摆脱输入向量z受输入数据集分布的影响，更好的实现属性解耦合
class MappingNetwork(torch.nn.Module):
    def __init__(self,z_dim,w_dim):
        super(MappingNetwork, self).__init__()
        self.mapping = torch.nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim,w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )
    def forward(self,x):
        return self.mapping(x)

#注入到网络中的噪声
class InjectNoise(torch.nn.Module):
    def __init__(self,channel):
        super(InjectNoise, self).__init__()
        #torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，
        # 并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可
        # 以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
        self.weight = torch.nn.Parameter(torch.zeros(1,channel,1,1))

    def forward(self,x):
        #产生的噪声，并且噪声的维度和x的维度是一样的
        noise = torch.randn(size = (x.shape[0],1,x.shape[2],x.shape[3]),device=x.device)
        #这里之所以使用self.weight * noise表示将noise设置为可训练的参数
        out = x + self.weight * noise
        return out

#是一个Batch Normazliation
class AdaIN(torch.nn.Module):
    def __init__(self,channel,w_dim):
        super(AdaIN, self).__init__()
        #https://blog.csdn.net/OneFlow_Official/article/details/123288435
        self.instance_norm = torch.nn.InstanceNorm2d(channel)
        self.style_scale = WSLinear(w_dim,channel)
        self.style_bias = WSLinear(w_dim,channel)

    def forward(self,x,w):
        x = self.instance_norm(x)
        #对dim = 2和dim = 3其进行升维
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


#对应着Synthesis network的每一个block
class GenBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels,out_channels)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels,w_dim)
        self.adain2 = AdaIN(out_channels,w_dim)

    def forward(self,x,w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))),w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))),w)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels,out_channels)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

#Generator的输入为4 x 4 x 512
class Generator(torch.nn.Module):
    def __init__(self,z_dim,w_dim,in_channels,img_channels = 3):
        super(Generator, self).__init__()
        #styleGAN的输入不依赖于噪声的输入，输入为一个常数
        self.starting_constant = torch.nn.Parameter(torch.ones(size=(1,in_channels,4,4)))
        self.map = MappingNetwork(z_dim,w_dim)
        self.inital_adain1 = AdaIN(in_channels,w_dim)
        self.inital_adain2 = AdaIN(in_channels,w_dim)
        #每个卷积之后添加一个噪声，通道为1，其中输入的B是可学习的
        self.inital_noise1 = InjectNoise(in_channels)
        self.inital_noise2 = InjectNoise(in_channels)

        self.inital_conv = torch.nn.Conv2d(in_channels,in_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)

        self.inital_rgb = WSConv2d(
            in_channels,img_channels,kernel_size=1,stride=1,padding=0
        )
        self.prog_blocks,self.rgb_layers = (
            torch.nn.ModuleList([]),
            torch.nn.ModuleList([self.inital_rgb])
        )
        #通道数的下降因子
        #factors = [1,1,1,1,1 / 2,1 / 4,1 / 8,1 / 16,1 / 32]
        #前面首先对输入的4 x 4 x 512的特征图进行了初始化操作
        #下面是经过8个GenBlock卷积块
        for i in range(len(factors) - 1):
            #输入的通道数
            conv_in_c = int(in_channels * factors[i])
            #输出的通道数
            conv_out_c = int(in_channels * factors[i + 1])
            #Synthesis network每一个块
            self.prog_blocks.append(GenBlock(conv_in_c,conv_out_c,w_dim))
            #最后采用1 x 1卷积输出的通道数为img_channels = 3
            self.rgb_layers.append(
                WSConv2d(conv_out_c,img_channels,kernel_size=1,stride=1,padding=0)
            )
    def fade_in(self,alpha,upscaled,generated):
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self,noise,alpha,steps):
        w = self.map(noise)
        x = self.inital_adain1(self.inital_noise1(self.starting_constant),w)
        x = self.inital_conv(x)
        out = self.inital_adain2(self.leaky(self.inital_noise2(x)),w)

        if steps == 0:
            return self.inital_rgb(x)

        upscaled = 0
        for step in range(steps):
            #采用spherical interpolation的采样方法
            upscaled = F.interpolate(out,scale_factor=2,mode='bilinear')
            out = self.prog_blocks[step](upscaled,w)

        # The number of channels in upscale will stay the same, while
        # out which has moved through prog_blocks might change. To ensure
        # we can convert both to rgb we use different rgb_layers
        # (steps-1) and steps for upscaled, out respectively
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha,final_upscaled,final_out)


class Discriminator(torch.nn.Module):
    def __init__(self,in_channels,img_channels = 3):
        super(Discriminator, self).__init__()
        self.prog_blocks,self.rgb_layers = torch.nn.ModuleList([]),torch.nn.ModuleList([])
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)

        # here we work back ways from factors because the discriminator
        # should be mirrored from the generator. So the first prog_block and
        # rgb layer we append will work for input size 1024x1024, then 512->256-> 128-> 64-> 32-> 16-> 4
        for i in range(len(factors) - 1,0,-1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in,conv_out))
            self.rgb_layers.append(
                WSConv2d(img_channels,conv_in,kernel_size=1,stride=1,padding=0)
            )
        # perhaps confusing name "initial_rgb" this is just the RGB layer for 4x4 input size
        # did this to "mirror" the generator initial_rgb
        self.inital_rgb = WSConv2d(
            img_channels,in_channels,kernel_size=1,stride=1,padding=0
        )
        self.rgb_layers.append(self.inital_rgb)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(2,2),stride=(2,2)
        )

        # this is the block for 4x4 input size
        self.final_block = torch.nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1,in_channels,kernel_size=3,padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            WSConv2d(in_channels , in_channels, kernel_size=4, stride=1,padding=0),
            torch.nn.LeakyReLU(negative_slope=0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1,padding=0),
        )
    def fade_in(self,alpha,downscaled,out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self,x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)
    def forward(self,x,alpha,steps):
        # where we should start in the list of prog_blocks, maybe a bit confusing but
        # the last is for the 4x4. So example let's say steps=1, then we should start
        # at the second to last because input_size will be 8x8. If steps==0 we just
        # use the final block
        cur_step = len(self.prog_blocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        # this is opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == '__main__':
    Z_DIM = 512
    W_DIM = 512
    IN_CHANNELS = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=3).to(device)
    disc = Discriminator(IN_CHANNELS, img_channels=3).to(device)

    tot = 0
    for param in gen.parameters():
        tot += param.numel()

    print('total parameters: {}'.format(tot))

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((2, Z_DIM)).to(device)
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (2, 3, img_size, img_size)
        out = disc(z, alpha=0.5, steps=num_steps)
        # print(out.shape)
        assert out.shape == (2, 1)
        print(f"Success! At img size: {img_size}")

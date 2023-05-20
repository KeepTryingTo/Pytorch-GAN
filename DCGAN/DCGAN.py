"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/26 15:06
"""

import time
import torch
import torchvision.datasets

import DCGen
import DCDis
import numpy as np
import matplotlib.pyplot as  plt
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir = 'logs')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#预处理图片
transform = transforms.Compose([
    transforms.Resize(size = (64,64)),
    transforms.CenterCrop(size = (64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5]),
])


#训练图片的路径
trainPath = 'data/'
#训练的批次
batchSize = 16

#加载数据集
trainDataset = torchvision.datasets.ImageFolder(
    root=trainPath,
    transform=transform
)
dataLoader = DataLoader(dataset=trainDataset,batch_size=batchSize,shuffle=True,num_workers=0)
print('dataSize : {}'.format(len(dataLoader.dataset)))

#加载生成器和判别器
gen = DCGen.Generator().to(device)
dis = DCDis.Discriminator().to(device)
# print('GenModel :{}'.format(gen))
# print('DisModel :{}'.format(dis))

#优化器选择
learningRate = 0.0001
betal = 0.5
d_Optimer = torch.optim.Adam(params=dis.parameters(), lr = learningRate, betas = (betal,0.9))
g_Optimer = torch.optim.Adam(params=gen.parameters(), lr = learningRate, betas = (betal,0.9))

#交叉熵损失函数
loss_Fn = torch.nn.BCELoss()

#绘图函数
def DrawGen(model,epoch,test_input):
    """
    :param model: 生成器训练的模型
    :param epoch: 迭代次数
    :param test_input: 对产生的噪声生成图像
    :return:
    """
    result = model(test_input).detach().cpu()
    #将维度为1的进行压缩
    #--------------------------------------------------------------
    # vutilsImg = vutils.make_grid(result, padding=2, normalize=True)
    # fig = plt.figure(figsize=(4, 4))
    # plt.imshow(np.transpose(vutilsImg, (1, 2, 0)))
    # plt.axis('off')
    # plt.show()
    # --------------------------------------------------------------
    result = np.squeeze(result.numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.transpose(result[i],(1,2,0)))
        plt.axis('off')
    plt.savefig('images/{}.png'.format(epoch))
    #--------------------------------------------------------------

#训练迭代的次数
epoches = 20
test_input = torch.randn(size = (16,100,1,1), device=device)

d_Loss=[]
g_Loss=[]

for epoch in range(epoches):
    d_epoch_loss = 0
    g_epoch_loss = 0
    lenDataset = len(dataLoader.dataset)
    for step,data in enumerate(dataLoader):
        imgs, _ =data
        real_Imgs = imgs.to(device)
        size = real_Imgs.shape
        fake_Imgs = torch.randn(size = (size[0],100,1,1),device=device)

        d_Optimer.zero_grad()
        real_ouput = dis(real_Imgs).view(-1)
        # 期望判别器判别真实图像结果为1 真实图像上的损失值
        d_real_loos = loss_Fn(real_ouput, torch.ones_like(real_ouput))
        # d_real_loos.backward()

        # 生成器上生成图片
        gen_img = gen(fake_Imgs)
        # 根据生成器生成的图片，判别器进行判断
        fake_output = dis(gen_img.detach()).view(-1)
        d_fake_loss = loss_Fn(fake_output, torch.zeros_like(fake_output))
        # d_fake_loss.backward()

        # 判别器损失值 = 真实图片上的损失值 + 假图像上的损失值
        d_loss = d_fake_loss + d_real_loos
        d_loss.backward()
        d_Optimer.step()

        # #对于生成器来说，期望判别器判定为真
        g_Optimer.zero_grad()
        fake_output = dis(gen_img).view(-1)
        g_loss = loss_Fn(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        g_Optimer.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
            if step % 1000 == 0 and step != 0:
                print('--------------------------------------{} steps / 1000------------------------------ '.format(
                    step / 1000))
                print('*****************      genLoss : {:.6f}   disLoss : {:.6f}       *****************'.format(
                    d_epoch_loss / step,
                    g_epoch_loss / step))
    with torch.no_grad():
        d_Loss.append(d_epoch_loss / lenDataset)
        g_Loss.append(g_epoch_loss / lenDataset)
        print('--------------------------------------{} epochs       ------------------------------ '.format(epoch))
        print('*****************      genLoss : {:.6f}   disLoss : {:.6f}      *****************'.format(
            d_epoch_loss / lenDataset,
            g_epoch_loss / lenDataset))
        DrawGen(gen, epoch, test_input=test_input)
        writer.add_scalar(tag = 'Gen',scalar_value=g_epoch_loss / lenDataset,global_step=epoch)
        writer.add_scalar(tag='Dis', scalar_value=d_epoch_loss / lenDataset, global_step=epoch)

# 保存生成器和判别器模型
torch.save(obj=dis, f='models/dis.pth')
torch.save(obj=gen, f='models/gen.pth')


writer.close()

if __name__ == '__main__':
    pass


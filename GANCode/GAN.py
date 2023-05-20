"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/24 14:01
"""
import torch
import gModel
import dModel
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

#加载数据集MINST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5],std = [0.5])#如果是RGB彩色图像，应该为mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]
])
trainDataset = torchvision.datasets.MNIST(root = 'data/mnist',train = True,transform = transform,download = True)
testDataset = torchvision.datasets.MNIST(root = 'data/mnist',train = False,transform = transform,download = True)

#对数据集进行打包
dataLoader = DataLoader(dataset = trainDataset,batch_size = 16,num_workers = 0,shuffle = True)

#查看数据集
imgs,labels = next(iter(dataLoader))
print('labels: {}'.format(labels))
print('imgs.shape: {}'.format(imgs.shape))

#labels: tensor([5, 0, 4, 1])
#imgs.shape: torch.Size([4, 1, 28, 28])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#导入模型
gen = gModel.Generator().to(device)
dis = dModel.Discriminator().to(device)

#选择优化器
learingRate = 0.0001
D_optimer = torch.optim.Adam(params = dis.parameters(),lr = learingRate)
G_opimer = torch.optim.Adam(params = gen.parameters(), lr = learingRate)

#损失函数BCELoss - 不包含sigmoid
loss_Fn = torch.nn.BCELoss()


#绘图函数
def DrawGen(model,epoch,test_input):
    """
    :param model: 生成器训练的模型
    :param epoch: 迭代次数
    :param test_input: 对产生的噪声生成图像
    :return:
    """
    result = model(test_input).detach().cpu().numpy()
    #将维度为1的进行压缩
    result = np.squeeze(result)
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        #由于生成器输出的结果为[-1,1]之间，所以需要转换为[0,1]之间
        plt.imshow((result[i] + 1) / 2)
        plt.axis('off')
    plt.savefig('images/{}.png'.format(epoch))
    # plt.show()


#训练生成器
DLoss = []
GLoss = []

epoches = 100

test_inut = torch.randn(size = (16, 100), device=device)

for epoch in range(epoches):

    d_epoch_loss = 0
    g_epoch_loss = 0

    lenDataset = len(dataLoader.dataset)

    for step ,data in enumerate(dataLoader):
        imgs,labels = data
        real_imgs = imgs.to(device)
        #获取图片的数量
        size = real_imgs.size(0)
        #产生相同图片数量的正态分布随机数(0 - 1)
        fake_imgs = torch.randn(size = (size , 100), device = device)

        D_optimer.zero_grad()
        real_ouput = dis(real_imgs)
        #期望判别器判别真实图像结果为1 真实图像上的损失值
        d_real_loos = loss_Fn(real_ouput,torch.ones_like(real_ouput))
        # d_real_loos.backward()

        #生成器上生成图片
        gen_img = gen(fake_imgs)
        #根据生成器生成的图片，判别器进行判断
        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_Fn(fake_output,torch.zeros_like(fake_output))
        # d_fake_loss.backward()

        #判别器损失值 = 真实图片上的损失值 + 假图像上的损失值
        d_loss = d_fake_loss + d_real_loos
        d_loss.backward()
        D_optimer.step()

        # #对于生成器来说，期望判别器判定为真
        G_opimer.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_Fn(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        G_opimer.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
            if step % 1000 == 0 and step != 0:
                print('--------------------------------------{} steps / 10000------------------------------ '.format(step / 10000))
                print('*****************      genLoss : {:.6f}   disLoss : {:.6f}       *****************'.format(d_epoch_loss / step,
                                                                   g_epoch_loss / step))
    with torch.no_grad():
        DLoss.append(d_epoch_loss / lenDataset)
        GLoss.append(g_epoch_loss / lenDataset)
        print('--------------------------------------{} epochs       ------------------------------ '.format(epoch))
        print('*****************      genLoss : {:.6f}   disLoss : {:.6f}      *****************'.format(d_epoch_loss / lenDataset,
                                                               g_epoch_loss / lenDataset))
        DrawGen(gen, epoch, test_input = test_inut)

#保存生成器和判别器模型
torch.save(obj = dis, f = 'models/dis.pth')
torch.save(obj = gen, f = 'models/gen.pth')

if __name__ == '__main__':
    pass


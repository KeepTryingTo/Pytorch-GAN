"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/19 10:01
"""

import os
import math
import torch
import config
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from utils import saveImage,save_checkpoint
from torch.utils.data import DataLoader
from models import Generator,Discriminator,weights_init_normal

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=config.NUM_EPOCHS, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='size of the batches')
    parser.add_argument('--lr', type=float, default=config.LEARING_RATIO, help='adam: learning rate')
    parser.add_argument('--beta1', type=float, default=config.BETA1, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=config.BETA2, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=config.NUM_WORKS, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=config.LATENT_DIM, help='dimensionality of the latent space')
    parser.add_argument('--n_classes', type=int, default=config.NUM_CLASSES, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=config.IMG_SIZE, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=config.CHANNELS, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=config.SAMPLE_INTERVAL, help='interval between image sampling')
    args = parser.parse_args()
    print('args: {}'.format(args))
    return args


def train_fn(generator,discriminator,optimizer_G,optimizer_D,adversarial_loss,dataloader,epoch):
    loop = tqdm(dataloader,leave=True)
    loop.set_description(desc="Training: ")
    for step,(imgs,labels) in enumerate(loop):
        valid = torch.ones(config.BATCH_SIZE).to(config.DEVICE)
        fake = torch.zeros(config.BATCH_SIZE).to(config.DEVICE)

        # print('labels.shape: {}'.format(np.shape(labels)))
        # print('labels: {}'.format(labels))

        real_imgs = imgs.to(config.DEVICE)
        real_y = torch.zeros(config.BATCH_SIZE,config.NUM_CLASSES)
        #dim = 1,index = config.BATCH_SIZE,src = 1
        #https://mbd.baidu.com/ma/s/HT3QuRvI
        real_y = real_y.scatter_(1, labels.view(config.BATCH_SIZE, 1), 1).view(config.BATCH_SIZE,config.NUM_CLASSES,1,1)
        real_y = real_y.expand(-1,-1,config.IMG_SIZE,config.IMG_SIZE).to(config.DEVICE)

        noise = torch.randn(size = (config.BATCH_SIZE,config.LATENT_DIM,1,1)).to(config.DEVICE)
        gen_labels = (torch.rand(config.BATCH_SIZE,1)*config.NUM_CLASSES).type(torch.LongTensor)

        # print('gen_labels.shape: {}'.format(gen_labels.shape)) => [BATCH_SIZE,1]
        # print('gen_labels: {}'.format(gen_labels))[[3.6908],[8.2607],[7.1017],......]

        gen_y = torch.zeros(config.BATCH_SIZE,config.NUM_CLASSES)
        # dim = 1,index = config.BATCH_SIZE,src = 1
        gen_y = gen_y.scatter_(1, gen_labels.view(config.BATCH_SIZE, 1), 1).view(config.BATCH_SIZE,config.NUM_CLASSES,1,1)
        gen_y = gen_y.to(config.DEVICE)

        #expand :https://blog.csdn.net/weixin_39504171/article/details/106090626/
        gen_y_for_D = gen_y.view(config.BATCH_SIZE, config.NUM_CLASSES, 1, 1).expand(-1, -1, config.IMG_SIZE, config.IMG_SIZE)
        gen_y_for_D = gen_y_for_D.to(config.DEVICE)

        #compute the discriminator's loss
        optimizer_D.zero_grad()
        d_real_loss = adversarial_loss(np.squeeze(discriminator(real_imgs,real_y)),valid)
        gen_imgs = generator(noise,gen_y)
        d_fake_loss = adversarial_loss(np.squeeze(discriminator(gen_imgs.detach(),gen_y_for_D)),fake)
        # d_fake_loss = adversarial_loss(np.squeeze(discriminator(gen_imgs.detach(), real_y)), fake)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        #compute the generator's loss
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(np.squeeze(discriminator(gen_imgs,gen_y_for_D)),valid)
        # g_loss = adversarial_loss(np.squeeze(discriminator(gen_imgs, real_y)), valid)
        g_loss.backward()
        optimizer_G.step()

        loop.set_postfix(
            epoch = epoch,
            d_loss = d_loss.item(),
            g_loss = g_loss.item()
        )



def main(args):
    #加载数据集
    dataset = torchvision.datasets.MNIST(root='data/',train=True,transform=config.transform,
                                         download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
        drop_last=config.DROP_LAST,

    )
    #加载模型
    generator = Generator().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    #定义损失函数和优化器
    adversarial_loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(),lr=config.LEARING_RATIO,betas=(config.BETA1,config.BETA2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr = config.LEARING_RATIO,betas=(config.BETA1,config.BETA2))

    for epoch in range(config.NUM_EPOCHS):
        train_fn(generator,discriminator,optimizer_G,optimizer_D,adversarial_loss,dataloader,epoch)
        if epoch % 10 == 0:
            saveImage(generator, epoch)
            save_checkpoint(generator,optimizer_G,config.CHECKPOINT_GEN)


if __name__ == '__main__':
    args = Parser()
    main(args)
    pass

"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/6 15:53
"""
import time
import torch
import config
import utils
import numpy as np
from tqdm import tqdm
import torchvision.datasets
from net.Decoder import Decoder
from net.Encoder import Encoder
from torchvision import transforms
from torch.utils.data import DataLoader
from net.Discriminator import Discrimiantor

#下载数据集
trainDataset = torchvision.datasets.FashionMNIST(root=config.TRAIN_DATA,train=True,
                                          transform=config.transform,download=True)
valDataset = torchvision.datasets.FashionMNIST(root=config.VAL_DATA,train=False,
                                          transform=config.transform,download=True)

#加载数据集
trainLoader = DataLoader(
    dataset=trainDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
valLoader = DataLoader(
    dataset=valDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

#加载模型
encoder = Encoder(in_features=784,out_features=128).to(config.DEVICE)
decoder = Decoder(in_features=784,out_features=128).to(config.DEVICE)
disc = Discrimiantor(in_features=128).to(config.DEVICE)

#定义损失函数和优化器
loss_AE = torch.nn.MSELoss()
loss_disc = torch.nn.BCELoss()
loss_en = torch.nn.BCELoss()

opt_AE = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr = config.LEARNING_RATE,betas=(0.9,0.999))
opt_en = torch.optim.Adam(encoder.parameters(),lr=config.LEARNING_RATE,betas=(0.9,0.999))
opt_disc = torch.optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.9,0.999))

#保存损失值
loss_ = []


for epoch in range(config.NUM_EPOCHS):
    step_loss_en = 0
    step_loss_AE = 0
    step_loss_disc = 0
    loss_AE_disc_en = 0
    dataLen = len(trainLoader)
    encoder.train()
    disc.train()
    loop = tqdm(trainLoader,leave=True)
    loop.set_description(desc="training: ")
    for step ,data in enumerate(loop):
        imgs,labels = data
        imgs,labels = imgs.to(config.DEVICE),labels.to(config.DEVICE)
        #----------------------------------------------------------------
        #train encoder and decoder
        z_en = encoder(imgs)
        z_fake = decoder(z_en)
        loss_ae = loss_AE(z_fake,imgs)
        opt_AE.zero_grad()
        loss_ae.backward()
        opt_AE.step()
        step_loss_AE += loss_ae.item()
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        #train discriminator
        z_size = np.shape(imgs)[0]
        z_real = torch.randn(size=(z_size,128)).to(config.DEVICE)
        z_en_fake = encoder(imgs).detach()
        discInput = torch.cat((z_real,z_en_fake),dim = 0)
        discLabel = torch.cat((torch.ones(z_size,1),torch.zeros(z_size,1)),dim = 0).to(config.DEVICE)

        discOutput = disc(discInput)
        loss_disc_out = loss_disc(discOutput,discLabel)
        opt_disc.zero_grad()
        loss_disc_out.backward()
        opt_disc.step()
        step_loss_disc += loss_disc_out.item()
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        #train encoder
        z_en = encoder(imgs).detach()
        enOutput = disc(z_en)
        loss_en_out = loss_en(enOutput,torch.ones(z_size,1).to(config.DEVICE))
        opt_en.zero_grad()
        loss_en_out.backward()
        opt_en.step()
        step_loss_en += loss_en_out.item()
        # ----------------------------------------------------------------

        loop.set_description(desc="training: ")

        loss_AE_disc_en += (step_loss_disc + step_loss_AE + step_loss_en) / 3
        if step % 10 == 0 and step > 0:
            loop.set_postfix(epoch = epoch,loss = loss_AE_disc_en)
            # print("---------------------------Loss: {:.6f}----------------------------".format(loss))
    loss_.append(loss_AE_disc_en / dataLen)
    print("---------------------------Loss: {:.6f}----------------------------".format(loss_AE_disc_en / dataLen))
    time.sleep(0.3)
    if epoch % 10 == 0 and epoch > 0:
        with torch.no_grad():
            loop = tqdm(iterable=valLoader,leave=True)
            for step,data in enumerate(loop):
                imgs,labels = data
                if step % 100 == 0 and step > 0:
                    utils.save_epoch_image(model1=encoder,model2=decoder,filename="images",val_loader=imgs,epoch = epoch)

utils.draw(loss_,config.NUM_EPOCHS)
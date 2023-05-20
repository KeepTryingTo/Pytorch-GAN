"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/4 13:28
"""

import torch
import config
import numpy as np
from tqdm import tqdm
from dataset import MapDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from net import Generator,Discriminator
from torchvision.utils import save_image
from utils import save_checkpoint,save_some_examples,load_checkpoint

def train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE):
    loop = tqdm(train_loader,leave=True)
    for idx,(x,y) in enumerate(loop):
        x,y = x.to(config.DEVICE),y.to(config.DEVICE)

        #train dsicriminator
        # with torch.cuda.amp.autocast(): x-对应的是卫星拍摄的真实图 y-表示对应卫星拍摄的Google map
        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x,y_fake.detach())
        D_real_loss = BCE(D_real,torch.ones_like(D_real))
        D_fake_loss = BCE(D_fake,torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        #train generator
        # with torch.cuda.amp.autocast():
        D_fake = disc(x,y_fake)
        G_fake_loss = BCE(D_fake,torch.ones_like(D_fake))
        L1 = L1_LOSS(y_fake,y)*config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            #设置进度条显示的信息，下面表示在显示过程中同时显示损失值
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )



def main():
    disc = Discriminator.Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator.Generator(in_channles=3).to(config.DEVICE)

    opt_disc = torch.optim.Adam(disc.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = torch.optim.Adam(gen.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))

    BCE = torch.nn.BCEWithLogitsLoss()
    L1_LOSS = torch.nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, gen, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    val_dataset = MapDataset(root_dir=config.VAL_DIR)

    train_loader = DataLoader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=True)

    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        # train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler)
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE)

        if config.SAVE_MODEL and epoch % 20 == 0:
            save_checkpoint(gen,opt_gen,config.CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,config.CHECKPOINT_DISC)
        save_some_examples(gen,val_loader,epoch,folder="images")


if __name__ == '__main__':
    main()
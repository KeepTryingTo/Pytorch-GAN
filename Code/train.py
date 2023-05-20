"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/28 17:14
"""
import os
import sys
import torch
import config
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import utils
from net.Generator import Generator
from datasets import vanGoPhotoDataset
from torchvision.utils import save_image
from net.Discriminator import Discriminator
from torch.utils.data import DataLoader,Dataset

def train_fn(disc_X,disc_Y,gen_G,gen_F,loader,opt_disc,opt_gen,L1,mse,d_scale,g_scale,epoch,cudaIsAvailable = False):
    loop = tqdm(loader,leave=True)
    for idx ,(vango,photo) in enumerate(loop):
        vango = vango.to(config.DEVICE)
        photo = photo.to(config.DEVICE)

        #train discriminator
       # if cudaIsAvailable==True:
        #    with torch.cuda.amp.autocast():
        #X -> Y
        fake_photo = gen_G(vango)
        D_X_real = disc_X(photo)
        D_X_fake = disc_X(fake_photo.detach())
        D_X_real_loss = mse(D_X_real,torch.ones_like(D_X_real))
        D_X_fake_loss = mse(D_X_fake, torch.zeros_like(D_X_fake))
        D_X_loss = D_X_fake_loss + D_X_real_loss

        #Y -> X
        fake_vango = gen_F(photo)
        D_Y_real = disc_Y(vango)
        D_Y_fake = disc_Y(fake_vango.detach())
        D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
        D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
        D_Y_loss = D_Y_fake_loss + D_Y_real_loss

        D_loss = D_X_loss + D_Y_loss

        if cudaIsAvailable:
            opt_disc.zero_grad()
            d_scale.scale(D_loss).backward()
            d_scale.step(opt_disc)
            d_scale.update()
        else:
            opt_disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

    #train Generator H and Z
    #with torch.cuda.amp.autocast():
        #adversarial loss for both generator
        D_X_fake = disc_X(fake_photo)
        D_Y_fake = disc_Y(fake_vango)
        loss_G_Y = mse(D_X_fake,torch.ones_like(D_X_fake))
        loss_G_X = mse(D_X_fake,torch.ones_like(D_Y_fake))

        #cycle loss
        cycle_vango = gen_F(fake_photo)
        cycle_photo = gen_G(fake_vango)
        cycle_vango_loss = L1(vango,cycle_vango)
        cycle_photo_loss = L1(photo,cycle_photo)

        # identity loss
        identity_vango = gen_F(vango)
        identity_photo = gen_G(photo)
        identity_vango_loss = L1(vango,identity_vango)
        identity_photo_loss = L1(photo,identity_photo)

        G_loss = (
            loss_G_X + loss_G_Y
            + cycle_vango_loss * config.LAMBDA_CYCLE
            +cycle_photo_loss * config.LAMBDA_CYCLE
            +identity_vango_loss * config.LAMBDA_IDENTITY
            +identity_photo_loss * config.LAMBDA_IDENTITY
        )
        if cudaIsAvailable:
            opt_gen.zero_grad()
            g_scale.scale(G_loss).backward()
            g_scale.step(opt_gen)
            g_scale.update()
        else:
            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()

        with torch.no_grad():
            if idx % 200 == 0:
                save_image(fake_photo*0.5 + 0.5,f"save_images/VangoTophoto{epoch}_{idx}.png")
                save_image(fake_vango*0.5 + 0.5,f"save_images/phtotToVango{epoch}_{idx}.png")
                print("******************************************************************\n")
                print("--------------------G_loss : {:.6}-------------------".format(G_loss))
                print("--------------------D_loss : {:.6}-------------------".format(D_loss))


def main_():
    disc_X = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Y = Discriminator(in_channels=3).to(config.DEVICE)
    gen_F = Generator(img_channels=3,num_residual=9).to(config.DEVICE)
    gen_G = Generator(img_channels=3,num_residual=9).to(config.DEVICE)
    opt_disc = torch.optim.Adam(
        list(disc_X.parameters()) + list(disc_Y.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5,0.999)
    )
    opt_gen = torch.optim.Adam(
        list(gen_G.parameters()) + list(gen_F.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5,0.999)
    )
    L1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    #导入预训练模型
    if config.LOAD_MODEL:
        utils.load_checkpoin(
            config.CHECKPOINT_GEN_H,gen_G,opt_gen,config.LEARNING_RATE
        )
        utils.load_checkpoin(
            config.CHECKPOINT_GEN_Z, gen_F, opt_gen, config.LEARNING_RATE
        )
        utils.load_checkpoin(
            config.CHECKPOINT_CRITICH_H, disc_X, opt_disc, config.LEARNING_RATE
        )
        utils.load_checkpoin(
            config.CHECKPOINT_CRITICH_Z, disc_Y, opt_disc, config.LEARNING_RATE
        )

    dataset = vanGoPhotoDataset(
        root_vango=config.TRAIN_DIR + "/trainA",root_photo=config.TRAIN_DIR + "/trainB",
        transform=config.transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    g_scale = 0
    d_scale = 0
    cudaIsAvailable = False
    if torch.cuda.is_available():
        g_scale = torch.cuda.amp.GradScaler()
        d_scale = torch.cuda.amp.GradScaler()
        cudaIsAvailable = True

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_X,disc_Y,gen_G,gen_F,loader,opt_disc,opt_gen,L1,mse,d_scale,g_scale,epoch,cudaIsAvailable)
        if config.SAVE_MODEL:
            utils.save_checkpoint(gen_G,opt_gen,filename=config.CHECKPOINT_GEN_H,epochs = epoch)
            utils.save_checkpoint(gen_F, opt_gen, filename=config.CHECKPOINT_GEN_Z,epochs = epoch)
            utils.save_checkpoint(disc_X, opt_disc, filename=config.CHECKPOINT_CRITICH_H,epochs = epoch)
            utils.save_checkpoint(disc_Y, opt_disc, filename=config.CHECKPOINT_CRITICH_Z,epochs = epoch)


if __name__ == '__main__':
    main_()

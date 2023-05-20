"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/17 16:45
"""

import torch
import config
import numpy as np
from tqdm import tqdm
import torchvision.datasets
from torchvision import transforms

import utils
from net.DenseVAE import Decoder,Encoder
from torch.utils.data import DataLoader,Dataset

transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

#加载MNIST数据集
trainDataset = torchvision.datasets.MNIST(root=config.DATASET_DIR,train=True,transform=transform,
                                          download=True)
testDataset = torchvision.datasets.MNIST(root=config.DATASET_DIR,train=False,transform=transform,
                                          download=True)

trainLoader = DataLoader(
    dataset=trainDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMEORY
)
testLoader = DataLoader(
    dataset=testDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMEORY
)

#加载模型
encoder = Encoder(hidden_dim=512,latent_dim=2).to(config.DEVICE)
decoder = Decoder(hidden_dim=256,latent_dim=2,num_features=784).to(config.DEVICE)

#定义优化器
opt_encoder = torch.optim.Adam(params=encoder.parameters(),lr=config.LEARNING_RATIO,
                               betas=(config.BEAT1,config.BEAT2),eps=config.EPSILON)
opt_decoder = torch.optim.Adam(params=decoder.parameters(),lr=config.LEARNING_RATIO,
                               betas=(config.BEAT1,config.BEAT2),eps=config.EPSILON)
opt_vae = torch.optim.Adam(params=list(decoder.parameters()) + list(encoder.parameters()),lr=config.LEARNING_RATIO,
                               betas=(config.BEAT1,config.BEAT2),eps=config.EPSILON)
loss_fn = torch.nn.MSELoss()

loss = []

for epoch in range(config.NUM_EPOCHS):
    e_d_loss = 0
    encoder.train()
    decoder.train()
    loop = tqdm(trainLoader,leave=True)
    loop.set_description(desc="training: ")
    for step,data in enumerate(loop):
        imgs,labels = data
        imgs,labels = imgs.to(config.DEVICE),labels.to(config.DEVICE)
        z_mean,z_log_var = encoder(imgs)
        z = utils.samples(args=[z_mean,z_log_var])
        output = decoder(z)
        vae_loss = utils.loss_fn(inputs=imgs,outputs=output,loss_fn=loss_fn,z_mean = z_mean,
                                 z_log_var=z_log_var,num_features=784)
        opt_vae.zero_grad()
        vae_loss.backward()
        opt_vae.step()

        e_d_loss += vae_loss.item()

        if step % 10 == 0 and step > 0:
            loop.set_postfix(epoch = epoch,vae_loss = vae_loss.item() * config.LOSS_RATIO)
            if step % 1000 == 0 and step > 0:
                print('\n-------------------------------------VAE_LOSS: {:.6f}------------------------------------'.format(
                    vae_loss.item() * config.LOSS_RATIO))
    loss.append(e_d_loss * config.LOSS_RATIO / len(trainLoader.dataset))

    decoder.eval()
    encoder.eval()
    loop = tqdm(testLoader, leave=True)
    loop.set_description(desc="testing: ")
    for step, data in enumerate(loop):
        imgs, labels = data
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        z_mean, z_log_var = encoder(imgs)
        z = utils.samples(args=[z_mean, z_log_var])
        output = decoder(z)
        vae_loss = utils.loss_fn(inputs=imgs, outputs=output,loss_fn=loss_fn,z_mean=z_mean,
                                 z_log_var=z_log_var, num_features=784)
        if step % 300 == 0 and step > 0:
            utils.save_images(output,epoch,step)
            # utils.plot_predictions(y_true=imgs,y_pred=output,step = step,epoch = epoch)
            loop.set_postfix(epoch=epoch, vae_loss=vae_loss.item() * config.LOSS_RATIO)
            print('\n-------------------------------------VAE_LOSS: {:.6f}------------------------------------'.format(
                vae_loss.item() * config.LOSS_RATIO))
utils.draw(loss)

if __name__ == '__main__':
    pass
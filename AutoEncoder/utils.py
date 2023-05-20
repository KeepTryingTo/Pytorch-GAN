"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/6 16:27
"""

import os
import torch
import config
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def save_epoch_image(model,filename,val_loader,epoch):
    imgs = val_loader
    encoder_Decoder_Img = model(imgs).detach().cpu().numpy()
    imgs = np.squeeze(val_loader.detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    result = np.squeeze(encoder_Decoder_Img)

    #save encoder and decoder image
    for i in range(config.BATCH_SIZE):
        plt.subplot(4,4,i + 1)
        plt.imshow((result[i] + 1) / 2)
        plt.axis("off")
    plt.savefig(os.path.join(filename,f"{epoch}_AE.png"))
    plt.close('all')  # 避免内存泄漏

    fig = plt.figure(figsize=(4, 4))
    #save original images
    for i in range(config.BATCH_SIZE):
        plt.subplot(4,4,i + 1)
        plt.imshow((imgs[i] + 1) / 2)
        plt.axis("off")

    plt.savefig(os.path.join(filename,f"{epoch}_input.png"))
    plt.close('all')  # 避免内存泄漏


def draw(loss,epochs):
    plt.plot(range(1,epochs + 1),loss,label = 'trainLoss')
    plt.legend()
    plt.title('AE-LOSS')
    plt.savefig('logs/figure.png')
    # plt.show()


def save_checkPoint(model,optimizer,filename):
    print("=========================================================> save_model...")
    checkpoint = {
        "state_dict":model,
        'optimizer':optimizer.state_dict()
    }
    torch.save(obj=checkpoint,f=filename)

def load_checkPoint(model,filename,optimizer,lr):
    print("=========>")
    checkpoint = torch.load(f=filename,map_location=config.BEATAS)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/16 9:30
"""

import os
import torch
import config
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image


def saveImage(generator,epoch,):
    noise = torch.randn(size = (16,config.LATENT_DIM)).to(config.DEVICE)
    # fixed labels
    y_ = (torch.rand(16, 1) * config.NUM_CLASSES).type(torch.LongTensor)
    y_fixed = torch.zeros(16, config.NUM_CLASSES)
    y_fixed = y_fixed.scatter_(1, y_.view(16, 1), 1)

    gen_imgs = generator(noise, y_fixed).view(-1, config.CHANNELS,config.IMG_SIZE,config.IMG_SIZE)

    save_image(gen_imgs.data, 'save' + '/%d.png' % (epoch))


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

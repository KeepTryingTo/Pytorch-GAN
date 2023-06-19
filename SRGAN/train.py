"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/16 9:30
"""
import torch
import config
from torch import nn
from tqdm import tqdm
from torch import optim
from loss import VGGLoss
from dataset import MyImageFolder
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from utils import load_checkpoint, save_checkpoint, plot_examples

torch.backends.cudnn.benchmark = True

def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss,epoch):

    loop = tqdm(loader, leave=True)
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # print('low_res.shape: {}'.format(low_res.shape))
        # print('high_res.shape: {}'.format(high_res.shape))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        # print('disc_fake.shape: {}'.format(disc_fake.shape))
        # print('disc_real.shape: {}'.format(disc_real.shape))
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        # l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        #0.006 = 1 / (w x h) = 1 / (14 x 14)
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx % 200 == 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            epoch = epoch,
            loss_critic=loss_disc.item(),
            loss_gen=gen_loss.item()
        )


def main():
    dataset = MyImageFolder(root_dir=config.DATASET)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    #加载模型
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    #定义优化器
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    #定义均分误差损失函数
    mse = nn.MSELoss()
    #定义交叉熵损失函数
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_PRE,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_PRE,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss,epoch)

        print('epoch: {}'.format(epoch))
        if config.SAVE_MODEL and epoch > 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()

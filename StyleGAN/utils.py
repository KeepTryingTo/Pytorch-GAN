"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/15 14:44
"""
import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
import warnings
from torchvision.utils import save_image

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 32 examples
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)
        #保存图像到images文件下
        save_image(img_grid_fake,os.path.join('images','fake_' + str(tensorboard_step) + '.png'))
        save_image(img_grid_real,os.path.join('images','real_' + str(tensorboard_step) + '.png'))


#这里沿用了GP-GAN中的公式
def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

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


#保存模型，通常的做法
def save_checkpoint(model, optimizer, filename="models/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


#加载模型
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    # Found this useful (thanks alexis-jacq):
    # https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3
    def __init__(self, gamma=0.99, save=True, save_frequency=100, save_filename="ema_weights.pth"):
        """
        Initialize the weight to which we will do the
        exponential moving average and the dictionary
        where we store the model parameters
        """
        self.gamma = gamma
        self.registered = {}
        self.save_filename = save_filename
        self.save_frequency = save_frequency
        self.count = 0

        if save_filename in os.listdir("."):
            self.registered = torch.load(self.save_filename)

        if not save:
            warnings.warn("Note that the exponential moving average weights will not be saved to a .pth file!")

    def register_weights(self, model):
        """
        Registers the weights of the model which will
        later be used when we take the moving average
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.registered[name] = param.clone().detach()

    def __call__(self, model):
        self.count += 1
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_weight = param.clone().detach() if name not in self.registered else self.gamma * param + (1 - self.gamma) * self.registered[name]
                self.registered[name] = new_weight

        if self.count % self.save_frequency == 0:
            self.save_ema_weights()

    def copy_weights_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.registered[name]

    def save_ema_weights(self):
        torch.save(self.registered, self.save_filename)
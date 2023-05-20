"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/17 17:19
"""

"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 15:26
"""
import os
import torch
import config
import numpy as np
import matplotlib.pyplot as plt

#保存模型
def save_model(model,optimizer,epoch):
    """
    :param model:
    :param epoch:
    :return:
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(config.SAVE_MODELS,str(epoch)+'gen.tar'))


def load_checkpoin(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_group:
        param_group["lr"] = lr

def generate_and_save_images(save_dir,features,gen,epoch):
    """
    :param save_dir:
    :param gen:
    :param epoch:
    :return:
    """
    predictions = gen(features)
    pass

def draw(loss):
    """
    :param gen_loss:
    :param disc_loss:
    :return:
    """
    plt.plot(range(1, len(loss) + 1), loss, label='VAELoss')
    plt.legend()
    plt.title('VAE-LOSS')
    plt.savefig('logs/figure.png')

def save_images(imgs,epoch,step):
    imgs = imgs.detach().cpu().numpy()
    imgs = np.squeeze(imgs)
    fig = plt.figure(figsize=(2,2))
    for i in range(4):
        plt.subplot(2,2,i + 1)
        plt.imshow((imgs[i] + 1) / 2)
        plt.axis("off")
    plt.savefig(os.path.join(config.SAVE_IMAGES,str(epoch)+"_"+str(step)+'.png'))
    print("\n================================>Saving images...")

def plot_predictions(y_true,y_pred,step,epoch):
    """
    :param y_true: 真实的图像
    :param y_pred: 网络输出的图像
    :return:
    """
    f,ax = plt.subplots(nrows=2,ncols=10,sharex=True,sharey=True)
    for i in range(10):
        ax[0][i].imshow(np.reshape(y_true[i],(28,28)),aspect='auto')
        ax[1][i].imshow(np.reshape(y_pred[i].detach().cpu().numpy(),(28,28)),aspect='auto')
    #tight_layout会自动调整子图参数，使之填充整个图像区域。
    # 这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分。
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_IMAGES,str(epoch)+"_"+str(step)+'.png'))

def samples(args):
    """
    :param args: 编码器产生的均值和噪声
    :return:
    """
    z_mean,z_log_var = args
    eps = torch.nn.init.normal_(z_log_var,mean=0.,std=1.0)
    z = z_mean + torch.exp(z_log_var / 2) * eps
    return z

def loss_fn(inputs,outputs,loss_fn,z_mean,z_log_var,num_features = 784):
    reconstruction_loss = loss_fn(outputs,inputs)
    reconstruction_loss = reconstruction_loss * num_features

    #计算KL散度损失值
    kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
    kl_loss = -0.5 * torch.sum(kl_loss,dim = -1)
    vae_loss = torch.mean(reconstruction_loss + kl_loss)

    return vae_loss
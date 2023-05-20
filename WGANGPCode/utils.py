"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/3 15:27
"""

import torch

def gradient_penality(critic,real,fake,device='cpu'):
    """
    :param critic: 判别器模型
    :param real: 真实样本
    :param fake: 生成的样本
    :param device: 设备CUP or GPU
    :return:
    """
    BATCH_SIZE,C,H,W = real.shape
    alpha = torch.randn(size=(BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images=real*alpha + fake*(1-alpha)

    #计算判别器输出
    mixed_scores = critic(interpolated_images)
    #求导
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim = 1)
    gradient_penality = torch.mean((gradient_norm - 1)**2)
    return gradient_penality

"""
torch.autograd.grad函数参数如下：https://blog.csdn.net/waitingwinter/article/details/105774720

inputs: 求导的自变量
outputs: 求导的因变量（需要求导的函数）
grad_outputs:  如果 outputs为标量，则grad_outputs=None,也就是说，可以不用写;  如果outputs 是向量，则此参数必须写
retain_graph:  True 则保留计算图， False则释放计算图
create_graph: 若要计算高阶导数(二阶以二阶以上)，则必须选为True
allow_unused: 允许输入变量不进入计算
"""

if __name__ == '__main__':
    pass
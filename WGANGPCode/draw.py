"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 12:39
"""

import numpy as np
import matplotlib.pyplot as plt

#绘图函数
def DrawGen(model,epoch,test_input):
    """
    :param model: 生成器训练的模型
    :param epoch: 迭代次数
    :param test_input: 对产生的噪声生成图像
    :return:
    """
    result = model(test_input).detach().cpu()
    #将维度为1的进行压缩
    #--------------------------------------------------------------
    # vutilsImg = vutils.make_grid(result, padding=2, normalize=True)
    # fig = plt.figure(figsize=(4, 4))
    # plt.imshow(np.transpose(vutilsImg, (1, 2, 0)))
    # plt.axis('off')
    # plt.show()
    # --------------------------------------------------------------
    result = np.squeeze(result.numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.transpose(result[i],(1,2,0)))
        plt.axis('off')
    plt.savefig('images/{}.png'.format(epoch))
    #--------------------------------------------------------------
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/14 10:36
"""

import os
import torch
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

root_dir = "data/FFHQ/iamgeSize1024x1024/faces_dataset_small"
files = os.listdir(root_dir)

def resize(file,size,folder_to_save):
    """
    :param file:
    :param size:
    :param folder_to_save:
    :return:
    """
    image = Image.open(os.path.join(root_dir , file)).resize(size=(size,size),resample=Image.LANCZOS)
    image.save(folder_to_save + file,quality = 100)

if __name__ == '__main__':
    """
    os.mkdir:创建一级目录
    os.makedirs：创建多级目录
    """
    # for img_size in [4,8,512,1024]:
    #     folder_name  = "data/FFHQ_"+str(img_size)+'x'+str(img_size)+'/images/'
    #     if not os.path.isdir(folder_name):
    #         os.makedirs(folder_name)
    #     data = [(file,img_size,folder_name) for file in files]
    #     pool = Pool()
    #     pool.starmap(resize,data)
    pass

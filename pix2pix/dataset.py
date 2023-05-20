"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/4 14:44
"""

import os
import torch
import config
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MapDataset(Dataset):
    def __init__(self,root_dir):
        super(MapDataset, self).__init__()

        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir,img_file)
        image = np.array(Image.open(img_path))
        #图片由两张图片拼接的大小为1200 x 1200大小，
        input_image = image[:,:600,:]
        target_image = image[:,600:,:]
        #图像增强
        augmentations = config.both_transform(
            image = input_image,
            image0 = target_image
        )
        input_image,target_image = augmentations["image"],augmentations["image0"]

        input_image = config.transform_only_input(image = input_image)["image"]
        target_image = config.transform_only_mask(image = target_image)["image"]

        return input_image,target_image